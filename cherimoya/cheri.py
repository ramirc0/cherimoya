# cheri.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
The Cheri block — Cherimoya's adaptation of the ConvNeXt block to genomics.

Each block performs a 3-tap dilated depthwise convolution fused with a
per-example layer normalization, followed by an MLP expansion path with a
learnable per-channel residual scale (gamma). The fused conv+norm operation
has two implementations: a Triton GPU kernel for CUDA tensors, and a pure
PyTorch fallback that is used on CPU and on platforms without Triton. Both
paths are numerically equivalent up to floating-point error.
"""

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
	import triton
	import triton.language as tl
	HAS_TRITON = True
except ImportError:
	HAS_TRITON = False


CONV_NORM_EPS = 1e-3


def _cheri_conv_norm_cpu(x, w, dilation, eps=CONV_NORM_EPS):
	"""Pure PyTorch implementation of the fused dilated conv + norm.

	Performs a 3-tap depthwise dilated convolution followed by a per-example
	layer normalization across the (length, channels) plane. This is the
	fallback used on CPU and on platforms where Triton is unavailable. It
	participates in autograd automatically because it is built from
	standard differentiable PyTorch ops.

	Parameters
	----------
	x: torch.Tensor, shape=(N, L, C)
		The input tensor.

	w: torch.Tensor, shape=(3, C)
		Depthwise convolution weights for the left, center, and right taps.

	dilation: int
		Spacing between the three taps. The kernel reads from positions
		(i - dilation, i, i + dilation) for each output position i, with
		zero padding outside the sequence.

	eps: float, optional
		Numerical-stability constant added to the variance before taking
		the reciprocal square root. Default is 1e-3.

	Returns
	-------
	y: torch.Tensor, shape=(N, L, C)
		The convolved and normalized output, in the same dtype as `x`.
	"""

	N, L, C = x.shape

	# F.conv1d expects (N, C, L) input and (C, 1, 3) weight for depthwise.
	weight = w.t().unsqueeze(1).contiguous()
	x_t = x.transpose(1, 2).contiguous()
	y_t = F.conv1d(x_t, weight, padding=dilation, dilation=dilation, groups=C)
	y = y_t.transpose(1, 2).contiguous()

	# Per-example layer norm across all (L, C) positions.
	flat = y.reshape(N, -1).float()
	mean = flat.mean(dim=1, keepdim=True)
	var = flat.var(dim=1, keepdim=True, unbiased=False)
	rstd = (var + eps).rsqrt()
	flat = (flat - mean) * rstd
	return flat.reshape(N, L, C).to(y.dtype)


if HAS_TRITON:

	def _autotune_configs():
		num_warps = [4, 8, 16]
		num_stages = [2, 3, 4, 5]
		configs = []

		for num_warp, num_stage in itertools.product(num_warps, num_stages):
			configs.append(triton.Config({
				'num_warps': num_warp,
				'num_stages': num_stage,
			}))

		return configs


	@triton.autotune(
		configs=_autotune_configs(),
		key=['C', 'L'],
		reset_to_zero=['Sum_ptr', 'Sq_sum_ptr']
	)
	@triton.jit
	def _fwd_stats_kernel(
		X_ptr, W_ptr, Y_ptr, Sum_ptr, Sq_sum_ptr,
		stride_xn, dilation,
		L: tl.constexpr,
		C: tl.constexpr,
		BLOCK_C: tl.constexpr,
		BLOCK_L: tl.constexpr
	):
		pid_n = tl.program_id(0)
		pid_l = tl.program_id(1)
		offs_c = tl.arange(0, BLOCK_C)[None, :]
		mask_c = offs_c < C

		w_idx = W_ptr + offs_c
		w0 = tl.load(w_idx,       mask=mask_c, other=0.0)
		w1 = tl.load(w_idx + C,   mask=mask_c, other=0.0)
		w2 = tl.load(w_idx + C*2, mask=mask_c, other=0.0)

		l_start = pid_l * BLOCK_L
		offs = l_start + tl.arange(0, BLOCK_L)[:, None]
		offs_l = offs - dilation
		offs_r = offs + dilation

		mask = (offs < L) & mask_c
		mask_l = (offs_l >= 0) & mask
		mask_r = (offs_r < L) & mask

		x_idx = X_ptr + pid_n * stride_xn + offs_c
		x1 = tl.load(x_idx + offs*C,   mask=mask,   other=0.0)
		x0 = tl.load(x_idx + offs_l*C, mask=mask_l, other=0.0)
		x2 = tl.load(x_idx + offs_r*C, mask=mask_r, other=0.0)

		conv = x0*w0 + x1*w1 + x2*w2

		y_idx = Y_ptr + pid_n * stride_xn + offs * C + offs_c
		tl.store(y_idx, conv, mask=mask)

		conv = conv.to(tl.float32)

		block_sum = tl.sum(conv)
		block_sq_sum = tl.sum(conv * conv)

		tl.atomic_add(Sum_ptr + pid_n, block_sum, sem='relaxed')
		tl.atomic_add(Sq_sum_ptr + pid_n, block_sq_sum, sem='relaxed')


	@triton.jit
	def _fwd_finalize_kernel(
		Sum_ptr, Sq_sum_ptr, Mean_ptr, Rstd_ptr,
		eps,
		L: tl.constexpr,
		C: tl.constexpr,
	):
		pid_n = tl.program_id(0)

		running_sum = tl.load(Sum_ptr + pid_n)
		running_sq_sum = tl.load(Sq_sum_ptr + pid_n)

		count = L * C
		mean = running_sum / count
		var = (running_sq_sum / count) - (mean * mean)
		rstd = 1.0 / tl.sqrt(var + eps)

		tl.store(Mean_ptr + pid_n, mean)
		tl.store(Rstd_ptr + pid_n, rstd)


	@triton.jit
	def _fwd_apply_kernel(
		Y_ptr, Mean_ptr, Rstd_ptr,
		stride_yn,
		L: tl.constexpr,
		C: tl.constexpr,
		BLOCK_C: tl.constexpr,
		BLOCK_L: tl.constexpr,
	):
		pid_n = tl.program_id(0)
		pid_l = tl.program_id(1)
		offs_c = tl.arange(0, BLOCK_C)[None, :]
		mask_c = offs_c < C

		mean = tl.load(Mean_ptr + pid_n)
		rstd = tl.load(Rstd_ptr + pid_n)

		l_start = pid_l * BLOCK_L
		offs = l_start + tl.arange(0, BLOCK_L)[:, None]
		mask = (offs < L) & mask_c

		y_idx = Y_ptr + pid_n * stride_yn + offs * C + offs_c
		conv = tl.load(y_idx, mask=mask, other=0.0).to(tl.float32)
		x_hat = (conv - mean) * rstd

		tl.store(y_idx, x_hat, mask=mask)


	@triton.autotune(
		configs=_autotune_configs(),
		key=['C', 'L'],
		reset_to_zero=['Sum_dy_ptr', 'Sum_dy_xhat_ptr']
	)
	@triton.jit
	def _bwd_stats_kernel(
		dY_ptr, X_ptr, W_ptr, Mean_ptr, Rstd_ptr,
		Conv_ptr, Sum_dy_ptr, Sum_dy_xhat_ptr,
		stride_xn, dilation,
		L: tl.constexpr,
		C: tl.constexpr,
		BLOCK_C: tl.constexpr,
		BLOCK_L: tl.constexpr
	):
		pid_n = tl.program_id(0)
		pid_l = tl.program_id(1)
		offs_c = tl.arange(0, BLOCK_C)[None, :]
		mask_c = offs_c < C

		w_idx = W_ptr + offs_c
		w0 = tl.load(w_idx,       mask=mask_c, other=0.0)
		w1 = tl.load(w_idx + C,   mask=mask_c, other=0.0)
		w2 = tl.load(w_idx + C*2, mask=mask_c, other=0.0)

		mean = tl.load(Mean_ptr + pid_n)
		rstd = tl.load(Rstd_ptr + pid_n)

		l_start = pid_l * BLOCK_L
		offs = l_start + tl.arange(0, BLOCK_L)[:, None]
		offs_l = offs - dilation
		offs_r = offs + dilation

		mask = (offs < L) & mask_c
		mask_l = (offs_l >= 0) & mask
		mask_r = (offs_r < L) & mask

		x_idx = X_ptr + pid_n * stride_xn + offs_c
		x0 = tl.load(x_idx + offs_l*C, mask=mask_l, other=0.0)
		x1 = tl.load(x_idx + offs*C,   mask=mask,   other=0.0)
		x2 = tl.load(x_idx + offs_r*C, mask=mask_r, other=0.0)

		conv = x0*w0 + x1*w1 + x2*w2

		conv_idx = Conv_ptr + pid_n * stride_xn + offs * C + offs_c
		tl.store(conv_idx, conv, mask=mask)

		x_hat = (conv.to(tl.float32) - mean) * rstd

		dy_idx = dY_ptr + pid_n * stride_xn + offs * C + offs_c
		dy = tl.load(dy_idx, mask=mask, other=0.0).to(tl.float32)

		tl.atomic_add(Sum_dy_ptr + pid_n, tl.sum(dy), sem='relaxed')
		tl.atomic_add(Sum_dy_xhat_ptr + pid_n, tl.sum(dy * x_hat), sem='relaxed')


	@triton.autotune(
		configs=_autotune_configs(),
		key=['C', 'L']
	)
	@triton.jit
	def _bwd_apply_kernel(
		dY_ptr, X_ptr, Mean_ptr, Rstd_ptr,
		Sum_dy_ptr, Sum_dy_xhat_ptr,
		Conv_ptr, dW_ptr,
		stride_xn, num_chunks, dilation,
		L: tl.constexpr,
		C: tl.constexpr,
		BLOCK_C: tl.constexpr,
		BLOCK_L: tl.constexpr
	):
		pid_n = tl.program_id(0)
		pid_l = tl.program_id(1)
		offs_c = tl.arange(0, BLOCK_C)[None, :]
		mask_c = offs_c < C

		mean = tl.load(Mean_ptr + pid_n)
		rstd = tl.load(Rstd_ptr + pid_n)

		sum_dy_val = tl.load(Sum_dy_ptr + pid_n)
		sum_dy_xhat_val = tl.load(Sum_dy_xhat_ptr + pid_n)
		count = L * C

		l_start = pid_l * BLOCK_L
		offs = l_start + tl.arange(0, BLOCK_L)[:, None]
		offs_l = offs - dilation
		offs_r = offs + dilation
		mask = (offs < L) & mask_c
		mask_l = (offs_l >= 0) & mask
		mask_r = (offs_r < L) & mask

		buf_idx = Conv_ptr + pid_n * stride_xn + offs * C + offs_c
		conv = tl.load(buf_idx, mask=mask, other=0.0).to(tl.float32)
		x_hat = (conv - mean) * rstd

		dy = tl.load(dY_ptr + pid_n * stride_xn + offs * C + offs_c, mask=mask, other=0.0).to(tl.float32)

		d_conv = (rstd / count) * (count * dy - sum_dy_val - x_hat * sum_dy_xhat_val)
		tl.store(buf_idx, d_conv, mask=mask)

		x_idx = X_ptr + pid_n * stride_xn + offs_c
		x0 = tl.load(x_idx + offs_l*C, mask=mask_l, other=0.0).to(tl.float32)
		x1 = tl.load(x_idx + offs*C,   mask=mask,   other=0.0).to(tl.float32)
		x2 = tl.load(x_idx + offs_r*C, mask=mask_r, other=0.0).to(tl.float32)

		dw0 = tl.sum(d_conv * x0, axis=0)[None, :]
		dw1 = tl.sum(d_conv * x1, axis=0)[None, :]
		dw2 = tl.sum(d_conv * x2, axis=0)[None, :]

		dw_idx = dW_ptr + (pid_n * num_chunks + pid_l) * (3 * C) + offs_c
		tl.store(dw_idx,         dw0, mask=mask_c)
		tl.store(dw_idx + C,     dw1, mask=mask_c)
		tl.store(dw_idx + 2 * C, dw2, mask=mask_c)


	@triton.autotune(
		configs=_autotune_configs(),
		key=['C', 'L']
	)
	@triton.jit
	def _bwd_dx_kernel(
		dConv_ptr, W_ptr, dX_ptr,
		stride_xn, dilation,
		L: tl.constexpr,
		C: tl.constexpr,
		BLOCK_C: tl.constexpr,
		BLOCK_L: tl.constexpr
	):
		pid_n = tl.program_id(0)
		pid_l = tl.program_id(1)
		offs_c = tl.arange(0, BLOCK_C)[None, :]
		mask_c = offs_c < C

		w_idx = W_ptr + offs_c
		w0 = tl.load(w_idx,       mask=mask_c, other=0.0).to(tl.float32)
		w1 = tl.load(w_idx + C,   mask=mask_c, other=0.0).to(tl.float32)
		w2 = tl.load(w_idx + C*2, mask=mask_c, other=0.0).to(tl.float32)

		l_start = pid_l * BLOCK_L
		offs = l_start + tl.arange(0, BLOCK_L)[:, None]
		offs_p = offs + dilation
		offs_m = offs - dilation
		mask = (offs < L) & mask_c

		dc_base = dConv_ptr + pid_n * stride_xn + offs_c
		dc_c = tl.load(dc_base + offs * C,   mask=mask,                 other=0.0).to(tl.float32)
		dc_p = tl.load(dc_base + offs_p * C, mask=(offs_p < L) & mask,  other=0.0).to(tl.float32)
		dc_m = tl.load(dc_base + offs_m * C, mask=(offs_m >= 0) & mask, other=0.0).to(tl.float32)

		# dx[p] = d_conv[p]*w1 + d_conv[p+d]*w0 + d_conv[p-d]*w2
		dx = dc_c * w1 + dc_p * w0 + dc_m * w2
		tl.store(dX_ptr + pid_n * stride_xn + offs * C + offs_c, dx, mask=mask)


	class FusedDilatedConvNormFunc(torch.autograd.Function):
		"""Triton-backed fused dilated convolution + per-example layer norm.

		Implements the same operation as :func:`_cheri_conv_norm_cpu` but
		uses a custom Triton kernel to fuse the convolution, statistics
		reduction, and normalization steps into a small number of GPU
		passes. Only callable on CUDA tensors.
		"""

		@staticmethod
		def forward(ctx, x, w, dilation):
			N, L, C = x.shape
			BLOCK_C = triton.next_power_of_2(C)
			BLOCK_L = 64
			eps = CONV_NORM_EPS

			NUM_PARTIALS = triton.cdiv(L, BLOCK_L)

			sum_buf = torch.zeros((N,), dtype=torch.float32, device=x.device)
			sq_sum_buf = torch.zeros((N,), dtype=torch.float32, device=x.device)

			mean = torch.empty((N,), dtype=torch.float32, device=x.device)
			rstd = torch.empty((N,), dtype=torch.float32, device=x.device)

			y = torch.empty_like(x)

			_fwd_stats_kernel[(N, NUM_PARTIALS)](
				x, w, y, sum_buf, sq_sum_buf,
				x.stride(0), dilation,
				L, C, BLOCK_C=BLOCK_C, BLOCK_L=BLOCK_L
			)

			_fwd_finalize_kernel[(N,)](
				sum_buf, sq_sum_buf, mean, rstd,
				eps,
				L, C,
			)

			_fwd_apply_kernel[(N, NUM_PARTIALS)](
				y, mean, rstd,
				y.stride(0),
				L, C, BLOCK_C=BLOCK_C, BLOCK_L=BLOCK_L
			)

			ctx.save_for_backward(x, w, mean, rstd)
			ctx.dilation = dilation
			return y

		@staticmethod
		def backward(ctx, dy):
			x, w, mean, rstd = ctx.saved_tensors
			N, L, C = x.shape
			BLOCK_C = triton.next_power_of_2(C)
			BLOCK_L = 64

			dy = dy.contiguous()

			NUM_CHUNKS = triton.cdiv(L, BLOCK_L)

			sum_dy = torch.zeros((N,), dtype=torch.float32, device=x.device)
			sum_dy_xhat = torch.zeros((N,), dtype=torch.float32, device=x.device)

			buf = torch.empty((N, L, C), dtype=torch.float32, device=x.device)
			dx = torch.empty_like(x)
			dw = torch.empty((N * NUM_CHUNKS, 3 * C), dtype=torch.float32, device=x.device)

			_bwd_stats_kernel[(N, NUM_CHUNKS)](
				dy, x, w, mean, rstd,
				buf, sum_dy, sum_dy_xhat,
				x.stride(0), ctx.dilation,
				L, C, BLOCK_C=BLOCK_C, BLOCK_L=BLOCK_L
			)

			_bwd_apply_kernel[(N, NUM_CHUNKS)](
				dy, x, mean, rstd,
				sum_dy, sum_dy_xhat,
				buf, dw,
				x.stride(0), NUM_CHUNKS, ctx.dilation,
				L, C, BLOCK_C=BLOCK_C, BLOCK_L=BLOCK_L
			)

			_bwd_dx_kernel[(N, NUM_CHUNKS)](
				buf, w, dx,
				x.stride(0), ctx.dilation,
				L, C, BLOCK_C=BLOCK_C, BLOCK_L=BLOCK_L
			)

			dw = dw.view(N * NUM_CHUNKS, 3, C).sum(dim=0)
			return dx.to(x.dtype), dw.to(x.dtype), None


def fused_dilated_conv_norm(x, w, dilation):
	"""Fused 3-tap dilated depthwise conv plus per-example layer norm.

	Dispatches to the Triton kernel when `x` is a CUDA tensor and Triton
	is available, otherwise to a pure-PyTorch fallback. The two
	implementations are numerically equivalent up to floating-point error.

	Parameters
	----------
	x: torch.Tensor, shape=(N, L, C)
		The input tensor.

	w: torch.Tensor, shape=(3, C)
		Depthwise convolution weights for the left, center, and right taps.

	dilation: int
		Spacing between the three taps.

	Returns
	-------
	y: torch.Tensor, shape=(N, L, C)
		The convolved and normalized output.
	"""

	if HAS_TRITON and x.is_cuda:
		return FusedDilatedConvNormFunc.apply(x, w, dilation)
	return _cheri_conv_norm_cpu(x, w, dilation)


class CheriBlock(torch.nn.Module):
	"""A single Cheri Block.

	The Cheri Block is the core building block of the Cherimoya model. It
	adapts the ConvNeXt block to noisy genomics data, with the goal of
	mixing spatial and channel information cheaply while remaining stable
	to train.

	The block performs the following operations on an input of shape
	(N, L, C):

	1. A 3-tap depthwise dilated convolution that mixes spatial information
	   independently for each channel.
	2. A per-example layer normalization across the (length, channel)
	   plane. The convolution and normalization are fused into one kernel.
	3. A pointwise expansion linear projection from C to ``expansion * C``
	   channels.
	4. A GELU non-linearity.
	5. A pointwise contraction linear projection back to C channels.
	6. A residual connection where the MLP output is scaled by a fixed
	   constant (``residual_scale``) before being added to the input. The
	   small constant keeps the residual path near-identity at
	   initialization, which stabilizes training of deep stacks.

	Parameters
	----------
	n_filters: int
		The number of channels (the C dimension).

	dilation: int
		Dilation rate for the depthwise convolution. The kernel reads
		from positions (i - dilation, i, i + dilation) at each output
		position i, with zero padding outside the sequence.

	expansion: int, optional
		The factor by which the inner MLP expands the channel dimension.
		The first projection maps ``n_filters -> expansion * n_filters``
		and the second projects back. Default is 2.

	residual_scale: float, optional
		Fixed scalar applied to the MLP output before it is added back
		to the residual stream. Default is 0.15.
	"""

	def __init__(self, n_filters, dilation, expansion=2, residual_scale=0.15):
		super().__init__()
		self.n_filters = n_filters
		self.dilation = dilation
		self.expansion = expansion
		self.residual_scale = residual_scale

		hidden = expansion * n_filters

		self.conv_weight = torch.nn.Parameter(torch.randn(3, n_filters))
		self.linear1 = torch.nn.Linear(n_filters, hidden, bias=False)
		self.linear2 = torch.nn.Linear(hidden, n_filters, bias=False)
		self.activation = torch.nn.GELU(approximate='tanh')

		torch.nn.init.trunc_normal_(self.conv_weight, std=0.02)
		torch.nn.init.trunc_normal_(self.linear1.weight, std=0.02)
		torch.nn.init.trunc_normal_(self.linear2.weight, std=0.02)

	def forward(self, X):
		"""Run the block on an input of shape (N, L, C)."""

		X_conv = fused_dilated_conv_norm(X, self.conv_weight, self.dilation)
		X_mlp = self.linear2(self.activation(self.linear1(X_conv)))
		return X + X_mlp * self.residual_scale
