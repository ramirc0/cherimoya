"""Tests for PeakNegativeSampler that don't require BigWig / FASTA files."""

import numpy
import pytest
import torch

from cherimoya.io import PeakNegativeSampler


def _make_sampler(n_peaks=8, n_negs=8, in_window=20, out_window=10,
		max_jitter=4, controls=False, **kwargs):
	# Allocate enough length for jitter on both sides.
	in_len = in_window + 2 * max_jitter
	out_len = out_window + 2 * max_jitter

	peak_seqs = torch.zeros(n_peaks, 4, in_len)
	# Mark each peak with a unique flag in channel 0 for traceability.
	for i in range(n_peaks):
		peak_seqs[i, 0, :] = float(i + 1)

	# Deterministic signal data so repeat calls to _make_sampler produce
	# bit-identical fixtures (otherwise determinism tests below would
	# fail because the signal tensors differ run-to-run).
	g = torch.Generator().manual_seed(12345)
	peak_signals = torch.randint(0, 5, (n_peaks, 1, out_len),
		generator=g).float()

	neg_seqs = torch.zeros(n_negs, 4, in_len)
	for i in range(n_negs):
		neg_seqs[i, 0, :] = -(float(i + 1))
	neg_signals = torch.randint(0, 5, (n_negs, 1, out_len),
		generator=g).float()

	peak_ctl = torch.zeros(n_peaks, 1, in_len) if controls else None
	neg_ctl = torch.zeros(n_negs, 1, in_len) if controls else None

	# Default the seed so tests are deterministic unless overridden.
	kwargs.setdefault('random_state', 0)

	sampler = PeakNegativeSampler(
		peak_sequences=peak_seqs,
		peak_signals=peak_signals,
		negative_sequences=neg_seqs,
		negative_signals=neg_signals,
		peak_controls=peak_ctl,
		negative_controls=neg_ctl,
		in_window=in_window,
		out_window=out_window,
		max_jitter=max_jitter,
		**kwargs,
	)
	return sampler


# --------- Basic shape / length contract ----------------------------------

def test_len_combines_peaks_and_negative_ratio():
	sampler = _make_sampler(n_peaks=10, n_negs=20, negative_ratio=0.1)
	# 10 peaks + 0.1 * 10 = 1 negatives => length 11.
	assert len(sampler) == 11


def test_getitem_returns_correct_shapes_no_controls():
	sampler = _make_sampler(in_window=20, out_window=10, max_jitter=4)
	X, y, label = sampler[0]
	assert X.shape == (4, 20)
	assert y.shape == (1, 10)
	assert label in (0, 1)


def test_getitem_returns_controls_tuple_when_provided():
	sampler = _make_sampler(controls=True, in_window=20, out_window=10,
		max_jitter=4)
	out = sampler[0]
	assert len(out) == 4
	X, X_ctl, y, label = out
	assert X.shape == (4, 20)
	assert X_ctl.shape == (1, 20)
	assert y.shape == (1, 10)


def test_peak_ordering_stable_within_an_epoch():
	"""peak_ordering must be set up in __init__ (so forked workers that
	never see idx==0 still have a valid ordering) and must stay constant
	for every monotonically-increasing index sequence within one epoch,
	including the very first call at idx==0."""

	sampler = _make_sampler(n_peaks=16, n_negs=4, negative_ratio=0,
		max_jitter=1)
	assert sampler.peak_ordering is not None
	assert sorted(sampler.peak_ordering.tolist()) == list(range(16))

	ordering_initial = sampler.peak_ordering.copy()
	for idx in range(6):  # includes idx==0
		sampler[idx]
		assert numpy.array_equal(sampler.peak_ordering, ordering_initial)


def test_getitem_works_without_idx_zero():
	"""With multiple DataLoader workers, worker N>=1 may never receive
	idx==0. __getitem__ must still produce a valid sample."""

	sampler = _make_sampler(n_peaks=8, n_negs=4, negative_ratio=0,
		max_jitter=2)
	X, y, label = sampler[5]
	assert label == 1
	assert X.shape[0] == 4


# --------- Per-epoch reshuffle (wrap-around detection) -------------------

def test_wrap_around_triggers_reshuffle():
	"""When idx jumps backward (an epoch boundary in any worker's index
	stream) the peak_ordering must be reshuffled."""

	sampler = _make_sampler(n_peaks=16, n_negs=4, negative_ratio=0,
		max_jitter=1)
	for idx in [3, 4, 5, 9]:
		sampler[idx]
	ordering_before_wrap = sampler.peak_ordering.copy()

	sampler[2]   # wrap-around
	assert not numpy.array_equal(sampler.peak_ordering, ordering_before_wrap)


def test_no_reshuffle_when_shuffle_false():
	"""shuffle=False disables peak-ordering shuffling at every epoch."""

	sampler = _make_sampler(n_peaks=16, n_negs=4, negative_ratio=0,
		max_jitter=1, shuffle=False)
	# peak_ordering should be the identity in every epoch.
	expected = numpy.arange(16)
	for idx in [3, 4, 5, 0, 1, 2]:
		sampler[idx]
		assert numpy.array_equal(sampler.peak_ordering, expected)


# --------- Determinism guarantee -----------------------------------------

def _draw_epoch(sampler, n=None):
	"""Return a list of (label, X[0,0], y_first) tuples for one epoch."""
	n = n if n is not None else len(sampler)
	out = []
	for i in range(n):
		X, y, label = sampler[i]
		# Use a small fingerprint so we don't compare large tensors.
		out.append((int(label), float(X[0, 0]), float(y[0, 0])))
	return out


def test_same_seed_produces_same_sequence():
	"""Two samplers built with the same random_state must produce
	identical __getitem__ outputs for every idx."""

	a = _make_sampler(n_peaks=20, n_negs=10, negative_ratio=0.5,
		max_jitter=2, random_state=42)
	b = _make_sampler(n_peaks=20, n_negs=10, negative_ratio=0.5,
		max_jitter=2, random_state=42)
	assert _draw_epoch(a) == _draw_epoch(b)


def test_different_seeds_produce_different_sequences():
	a = _make_sampler(n_peaks=20, n_negs=10, negative_ratio=0.5,
		max_jitter=2, random_state=1)
	b = _make_sampler(n_peaks=20, n_negs=10, negative_ratio=0.5,
		max_jitter=2, random_state=2)
	assert _draw_epoch(a) != _draw_epoch(b)


def test_getitem_is_pure_within_an_epoch():
	"""__getitem__(idx) within a single epoch is independent of call
	history — multiple calls with the same idx return the same data."""

	sampler = _make_sampler(n_peaks=16, n_negs=8, negative_ratio=0.5,
		max_jitter=2, random_state=5)
	first = sampler[3]
	# Visit some other indices, all monotonically increasing so we don't
	# wrap into a new epoch.
	sampler[5]; sampler[7]
	# Now go back. This will wrap and prepare the next epoch — the value
	# at idx=3 in the *next* epoch may differ. Compare within the same
	# epoch instead by re-indexing forward.
	second = sampler[8]
	again_at_3 = sampler[3]   # wraps to a new epoch
	# In the same epoch (epoch 0), the idx=3 result must reproduce when
	# we rebuild a fresh sampler.
	fresh = _make_sampler(n_peaks=16, n_negs=8, negative_ratio=0.5,
		max_jitter=2, random_state=5)
	again_in_fresh_epoch_0 = fresh[3]
	assert torch.equal(first[0], again_in_fresh_epoch_0[0])
	assert torch.equal(first[1], again_in_fresh_epoch_0[1])
	assert first[2] == again_in_fresh_epoch_0[2]
	# And the post-wrap reindex sees epoch 1 content, which differs from
	# epoch 0 with high probability.
	assert (
		not torch.equal(again_at_3[0], first[0])
		or not torch.equal(again_at_3[1], first[1])
	)


def test_each_peak_drawn_exactly_once_per_epoch():
	"""With deterministic positions, every peak appears exactly once
	per epoch and every negative draw comes from the negative tensor —
	this catches any mismatch between _labels and _source_idx."""

	sampler = _make_sampler(n_peaks=32, n_negs=16, negative_ratio=0.5,
		max_jitter=0, random_state=0)
	peaks_drawn, negatives_drawn = [], []
	for i in range(len(sampler)):
		X, _, label = sampler[i]
		marker = int(X[0, 0].item())
		# _make_sampler bakes a positive marker into peak rows and a
		# negative marker into negative rows.
		if label == 1:
			peaks_drawn.append(marker)
		else:
			negatives_drawn.append(marker)
	assert sorted(peaks_drawn) == list(range(1, 33))
	# 16 negative draws, every marker in [-16, -1].
	assert len(negatives_drawn) == 16
	assert all(-16 <= m <= -1 for m in negatives_drawn)


def test_label_proportions_are_exact():
	"""Exactly n_peaks slots are peak draws; the rest are negative."""

	sampler = _make_sampler(n_peaks=64, n_negs=64, negative_ratio=1.0,
		max_jitter=2)
	labels = [sampler[i][-1] for i in range(len(sampler))]
	assert labels.count(1) == 64
	assert labels.count(0) == 64


# --------- Equivalence between num_workers=1 and num_workers>1 -----------

def _build_minimal_dataloader(num_workers, n_peaks=16, batch_size=2,
		seed=0):
	sampler = _make_sampler(n_peaks=n_peaks, n_negs=8, max_jitter=1,
		negative_ratio=0.5, random_state=seed)
	loader = torch.utils.data.DataLoader(
		sampler, batch_size=batch_size, num_workers=num_workers,
		persistent_workers=num_workers > 0,
	)
	return loader


def test_dataloader_two_epochs_two_workers():
	"""Multi-worker loader yields the full dataset, every epoch."""
	loader = _build_minimal_dataloader(num_workers=2)
	for _ in range(2):
		count = sum(batch[0].shape[0] for batch in loader)
		assert count == len(loader.dataset)


def _flatten_batches(loader, n_epochs=1):
	"""Concatenate every batch from `loader` across `n_epochs` into one
	tensor stack so two loaders' outputs can be compared elementwise."""
	rows = []
	for _ in range(n_epochs):
		for batch in loader:
			X, y, label = batch
			for i in range(X.shape[0]):
				rows.append((float(X[i, 0, 0]), float(y[i, 0, 0]),
					int(label[i])))
	return rows


def test_num_workers_does_not_change_data_sequence():
	"""The headline guarantee: num_workers=1 and num_workers=4 must
	produce identical batch sequences for the same seed."""

	loader_a = _build_minimal_dataloader(num_workers=1, seed=99,
		n_peaks=24, batch_size=4)
	loader_b = _build_minimal_dataloader(num_workers=4, seed=99,
		n_peaks=24, batch_size=4)
	a_rows = _flatten_batches(loader_a, n_epochs=2)
	b_rows = _flatten_batches(loader_b, n_epochs=2)
	assert a_rows == b_rows


def test_num_workers_zero_matches_one_worker():
	"""num_workers=0 (in-process) must also match num_workers>=1."""

	loader_a = _build_minimal_dataloader(num_workers=0, seed=7,
		n_peaks=20, batch_size=4)
	loader_b = _build_minimal_dataloader(num_workers=2, seed=7,
		n_peaks=20, batch_size=4)
	assert _flatten_batches(loader_a) == _flatten_batches(loader_b)


# --------- Edge cases: negative_ratio=0 and max_jitter=0 -----------------

def test_negative_ratio_zero_excludes_negatives():
	"""With negative_ratio=0, length excludes negatives and every draw is
	a peak."""
	sampler = _make_sampler(n_peaks=10, n_negs=20, negative_ratio=0)
	assert len(sampler) == 10
	for i in range(len(sampler)):
		_, _, label = sampler[i]
		assert label == 1


def test_negative_ratio_negative_raises():
	with pytest.raises(ValueError, match="negative_ratio"):
		_make_sampler(negative_ratio=-0.1)


def test_max_jitter_zero_returns_full_slice():
	"""max_jitter=0 must not crash the RNG and must return the unjittered
	(offset 0) window."""
	sampler = _make_sampler(n_peaks=2, n_negs=2, max_jitter=0,
		in_window=12, out_window=6, negative_ratio=0)
	for i in range(len(sampler)):
		X, y, label = sampler[i]
		assert label == 1
		assert X.shape == (4, 12)
		assert y.shape == (1, 6)
		# The whole peak row was filled with the same marker, so the
		# offset-0 slice should be uniformly that value.
		marker = X[0, 0].item()
		assert (X[0] == marker).all()


def test_max_jitter_negative_raises():
	with pytest.raises(ValueError, match="max_jitter"):
		_make_sampler(max_jitter=-1)


def test_reverse_complement_flips_when_enabled():
	"""With reverse_complement=True roughly half the peak draws should
	have been flipped along the channel/length axes."""

	sampler = _make_sampler(n_peaks=200, n_negs=20, in_window=8,
		out_window=4, max_jitter=1, reverse_complement=True,
		negative_ratio=0)
	flipped = 0
	peak_draws = 0
	for i in range(len(sampler)):
		X, _, label = sampler[i]
		if label != 1:
			continue
		peak_draws += 1
		# Each unflipped peak has a positive marker in channel 0; flipping
		# along (0, 1) puts a zero there.
		if X[0, 0].item() == 0.0:
			flipped += 1
	assert peak_draws == 200
	assert 0.4 * peak_draws < flipped < 0.6 * peak_draws
