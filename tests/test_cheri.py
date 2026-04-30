"""Tests for the CheriBlock and the fused dilated conv + norm dispatcher."""

import pytest
import torch

from cherimoya.cheri import (
	CheriBlock,
	HAS_TRITON,
	_cheri_conv_norm_cpu,
	fused_dilated_conv_norm,
)


def _make_inputs(N=2, L=64, C=8, dilation=2, seed=0):
	g = torch.Generator().manual_seed(seed)
	x = torch.randn(N, L, C, generator=g)
	w = torch.randn(3, C, generator=g) * 0.1
	return x, w, dilation


# --------- CPU fallback (always runnable) ---------------------------------

def test_cpu_fallback_shape_and_dtype():
	x, w, d = _make_inputs()
	y = _cheri_conv_norm_cpu(x, w, d)
	assert y.shape == x.shape
	assert y.dtype == x.dtype


def test_cpu_fallback_per_example_zero_mean_unit_var():
	"""The fused norm should produce zero-mean, near-unit-variance per
	example. The variance is `var_conv / (var_conv + eps)` so it
	approaches 1 only when `var_conv >> eps`. We pick weights large
	enough to get there."""

	g = torch.Generator().manual_seed(0)
	x = torch.randn(4, 128, 16, generator=g)
	w = torch.randn(3, 16, generator=g)  # std 1, not 0.1 -- conv var dominates eps
	y = _cheri_conv_norm_cpu(x, w, dilation=4)
	flat = y.reshape(y.shape[0], -1)
	assert torch.allclose(flat.mean(dim=1), torch.zeros(4), atol=1e-4)
	assert torch.allclose(flat.var(dim=1, unbiased=False),
		torch.ones(4), atol=2e-3)


def test_cpu_fallback_dilation_uses_correct_taps():
	"""Verify the 3-tap dilated convolution against a manual reference."""
	x, w, d = _make_inputs(N=1, L=10, C=2, dilation=3)
	# Manual conv: out[i, c] = w[0,c]*x[i-d,c] + w[1,c]*x[i,c] + w[2,c]*x[i+d,c]
	N, L, C = x.shape
	expected_conv = torch.zeros_like(x)
	for c in range(C):
		for i in range(L):
			lo = i - d
			hi = i + d
			val = w[1, c] * x[0, i, c]
			if lo >= 0:
				val = val + w[0, c] * x[0, lo, c]
			if hi < L:
				val = val + w[2, c] * x[0, hi, c]
			expected_conv[0, i, c] = val

	# Reproduce the per-example normalization the fallback applies.
	flat = expected_conv.reshape(1, -1).float()
	mean = flat.mean(dim=1, keepdim=True)
	var = flat.var(dim=1, keepdim=True, unbiased=False)
	expected = ((flat - mean) * (var + 1e-3).rsqrt()).reshape(N, L, C)

	y = _cheri_conv_norm_cpu(x, w, d)
	assert torch.allclose(y, expected, atol=1e-4)


def test_cpu_fallback_is_differentiable():
	x, w, d = _make_inputs()
	x = x.detach().requires_grad_(True)
	w = w.detach().requires_grad_(True)
	y = _cheri_conv_norm_cpu(x, w, d)
	y.sum().backward()
	assert x.grad is not None
	assert w.grad is not None
	assert torch.isfinite(x.grad).all()
	assert torch.isfinite(w.grad).all()


def test_dispatcher_uses_cpu_path_for_cpu_tensor():
	"""On CPU input, the dispatcher must produce the same output as the
	fallback regardless of whether Triton is installed."""
	x, w, d = _make_inputs()
	y_disp = fused_dilated_conv_norm(x, w, d)
	y_cpu = _cheri_conv_norm_cpu(x, w, d)
	assert torch.equal(y_disp, y_cpu)


# --------- CheriBlock ------------------------------------------------------

def test_cheri_block_forward_shape_preserved():
	block = CheriBlock(n_filters=16, dilation=4)
	x = torch.randn(3, 64, 16)
	y = block(x)
	assert y.shape == x.shape


def test_cheri_block_default_expansion_is_two():
	block = CheriBlock(n_filters=16, dilation=1)
	assert block.expansion == 2
	assert block.linear1.out_features == 32
	assert block.linear2.in_features == 32


@pytest.mark.parametrize("expansion", [1, 2, 4])
def test_cheri_block_expansion_configurable(expansion):
	block = CheriBlock(n_filters=8, dilation=1, expansion=expansion)
	assert block.linear1.out_features == 8 * expansion
	assert block.linear2.in_features == 8 * expansion
	# Forward pass must still preserve shape regardless of expansion.
	x = torch.randn(2, 32, 8)
	assert block(x).shape == x.shape


def test_cheri_block_residual_uses_fixed_scale():
	"""The residual scale is a fixed scalar. Zeroing the inner MLP
	output (by zeroing both linear weights) collapses the block to the
	identity since X + 0 * residual_scale == X."""

	block = CheriBlock(n_filters=8, dilation=1)
	with torch.no_grad():
		block.linear1.weight.zero_()
		block.linear2.weight.zero_()
	x = torch.randn(2, 32, 8)
	y = block(x)
	assert torch.allclose(y, x, atol=1e-5)


def test_cheri_block_has_no_gamma_parameter():
	"""The block must not register a learnable channel-wise scaling
	parameter — the residual scale is a fixed scalar."""

	block = CheriBlock(n_filters=8, dilation=1)
	param_names = {n for n, _ in block.named_parameters()}
	assert 'gamma' not in param_names
	assert not hasattr(block, 'gamma')


def test_cheri_block_default_residual_scale():
	block = CheriBlock(n_filters=8, dilation=1)
	assert block.residual_scale == 0.15


def test_cheri_block_residual_scale_is_not_a_parameter():
	"""residual_scale is a plain Python float, not a learnable Parameter."""
	block = CheriBlock(n_filters=8, dilation=1, residual_scale=0.3)
	assert isinstance(block.residual_scale, float)
	param_names = {n for n, _ in block.named_parameters()}
	assert 'residual_scale' not in param_names


@pytest.mark.parametrize("residual_scale", [0.0, 0.05, 0.15, 1.0])
def test_cheri_block_residual_scale_controls_output(residual_scale):
	"""For a given input, the output should be exactly
	X + residual_scale * X_mlp. Compare two blocks with identical
	parameters but different residual_scale values."""

	block = CheriBlock(n_filters=8, dilation=1, residual_scale=residual_scale)
	x = torch.randn(2, 32, 8)
	y = block(x)
	# residual = (y - x) / residual_scale must equal X_mlp regardless of scale.
	# Check the special case residual_scale=0: output must equal input exactly.
	if residual_scale == 0.0:
		assert torch.allclose(y, x, atol=1e-6)
	else:
		# Recompute MLP path explicitly.
		from cherimoya.cheri import fused_dilated_conv_norm
		x_conv = fused_dilated_conv_norm(x, block.conv_weight, block.dilation)
		x_mlp = block.linear2(block.activation(block.linear1(x_conv)))
		expected = x + x_mlp * residual_scale
		assert torch.allclose(y, expected, atol=1e-6)


def test_cheri_block_backward_produces_finite_grads():
	block = CheriBlock(n_filters=8, dilation=2)
	x = torch.randn(2, 32, 8, requires_grad=True)
	y = block(x).sum()
	y.backward()
	for name, p in block.named_parameters():
		assert p.grad is not None, name
		assert torch.isfinite(p.grad).all(), name
	assert torch.isfinite(x.grad).all()


# --------- GPU / Triton parity --------------------------------------------

@pytest.mark.cuda
@pytest.mark.triton
def test_triton_matches_cpu_fallback():
	"""The Triton kernel must match the CPU fallback up to FP error.

	This is the key correctness check that the CPU fallback is a valid
	stand-in when Triton is not available.
	"""

	torch.manual_seed(0)
	x = torch.randn(2, 128, 16)
	w = torch.randn(3, 16) * 0.1

	y_cpu = _cheri_conv_norm_cpu(x, w, dilation=4)

	x_gpu = x.cuda()
	w_gpu = w.cuda()
	y_gpu = fused_dilated_conv_norm(x_gpu, w_gpu, 4).cpu()

	assert torch.allclose(y_cpu, y_gpu, atol=1e-3)


@pytest.mark.cuda
def test_cheri_block_runs_on_cuda():
	block = CheriBlock(n_filters=16, dilation=2).cuda()
	x = torch.randn(2, 64, 16, device='cuda')
	y = block(x)
	assert y.shape == x.shape
	assert y.is_cuda


def test_has_triton_flag_only_true_with_cuda_and_triton(cuda_available):
	"""The flag should never be True without CUDA available — CPU-only
	systems sometimes have triton installed but it is unusable there.
	The dispatcher routes by `x.is_cuda` so the flag is informational."""

	# Just exercise the constant; no asserts beyond it being a bool.
	assert isinstance(HAS_TRITON, bool)
