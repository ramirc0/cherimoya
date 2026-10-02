# This file is named for cherimoya_cli/commands/fit.py. It checks that
# parameters reach the right downstream calls, with data loading and training
# faked, and the evaluate step that runs after training.
"""Wiring tests for `cherimoya fit` — confirms parameters flow into the
right downstream calls without actually training."""

from unittest import mock

import pytest
from omegaconf import OmegaConf


def _fit_config(**overrides):
	"""A fit config with fake inputs and every required key set."""
	from cherimoya_cli.config import FitConfig

	values = dict(sequences='fake.fa', loci='fake.bed',
		negatives='fake_negatives.bed', signals=['fake.bw'],
		name='fit_wiring_test', device='cpu')
	values.update(overrides)
	return OmegaConf.structured(FitConfig(**values))


@pytest.fixture
def fit_cfg():
	# The loader settings test_fit_forwards_the_loader_settings_* verifies.
	return _fit_config(num_workers=3, batch_size=16)


class _FakeDataset:
	"""Stands in for the `PeakNegativeSampler` behind `PeakGenerator`."""

	peak_sequences = __import__('torch').zeros(4, 4, 16)
	negative_sequences = __import__('torch').zeros(4, 4, 16)

	def __len__(self):
		return 8


class _FakeLoader(list):
	dataset = _FakeDataset()


def _run_capturing_training_fit(cfg, n_on_chroms=1):
	"""Run `fit.run` up to the training call and return its arguments.

	Data loading is faked and `cherimoya.training.fit` raises once it has
	recorded what it was given, so nothing trains.
	"""

	import torch

	from cherimoya_cli.commands import fit as fit_cmd

	captured = {}

	class _StopFit(Exception):
		pass

	def fake_fit(model, training_data, *args, **kwargs):
		captured.update(kwargs, model=model, training_data=training_data,
			valid=args)
		raise _StopFit()

	def fake_peak_generator(**kwargs):
		captured['peak_generator'] = kwargs
		return _FakeLoader([None] * 4)

	# One validation peak, and two validation negatives.
	def fake_extract_loci(**kwargs):
		captured.setdefault('extract', {})[kwargs['loci']] = kwargs
		n = 2 if kwargs['loci'] == 'fake_negatives.bed' else 1
		return torch.zeros(n, 4, 16), torch.zeros(n, 1, 8)

	with mock.patch("cherimoya.io.PeakGenerator",
				side_effect=fake_peak_generator), \
			mock.patch("tangermeme.io._interleave_loci",
				return_value=[None] * n_on_chroms), \
			mock.patch("tangermeme.io.extract_loci",
				side_effect=fake_extract_loci), \
			mock.patch("cherimoya.training.fit", side_effect=fake_fit):
		with pytest.raises(_StopFit):
			fit_cmd.run(cfg)

	return captured


def test_fit_forwards_the_loader_settings_to_training(fit_cfg):
	"""`num_workers` and `batch_size` go to the training module, which
	builds the loader, and the dataset rather than PeakGenerator's loader
	is what it receives."""

	captured = _run_capturing_training_fit(fit_cfg)

	assert captured['num_workers'] == 3
	assert captured['batch_size'] == 16
	assert isinstance(captured['training_data'], _FakeDataset)


@pytest.mark.parametrize("negatives,labels", [("fake_negatives.bed",
	[1, 0, 0]), (None, None)])
def test_fit_validates_on_the_negatives_too(fit_cfg, negatives, labels):
	"""The validation negatives follow the peaks and are labeled 0."""

	fit_cfg.negatives = negatives

	captured = _run_capturing_training_fit(fit_cfg)
	X_valid, y_valid = captured['valid']
	assert len(X_valid) == len(y_valid) == len(labels or [1])
	if labels is None:
		assert captured['labels_valid'] is None
	else:
		assert captured['labels_valid'].tolist() == labels


@pytest.mark.parametrize("device,accelerator", [("cpu", "cpu"),
	("cuda", "gpu")])
def test_fit_maps_device_to_a_lightning_accelerator(fit_cfg, device,
	accelerator):
	fit_cfg.device = device
	fit_cfg.devices = 2

	captured = _run_capturing_training_fit(fit_cfg)
	assert captured['accelerator'] == accelerator
	assert captured['devices'] == 2


@pytest.mark.parametrize("value", [None, False, True])
def test_fit_forwards_progress_bar(fit_cfg, value):
	fit_cfg.progress_bar = value

	assert _run_capturing_training_fit(fit_cfg)['progress_bar'] is value


def test_fit_schedules_count_the_partial_batch(fit_cfg):
	"""8 examples in batches of 16 is one step per epoch for the schedule,
	as a DataLoader counts it, with the default 2 warmup epochs and the
	20,000-step floor stretching the run to 20,000 epochs."""

	captured = _run_capturing_training_fit(fit_cfg)
	assert captured['n_warmup_steps'] == 2
	assert captured['max_epochs'] == 20000
	assert captured['n_decay_steps'] == 20000 - 2


def test_fit_launched_ranks_inherit_the_drawn_seed(fit_cfg, monkeypatch):
	"""A rank Lightning launches after the first re-runs `fit`, and must
	use the seed rank 0 drew rather than drawing its own."""

	fit_cfg.random_state = None

	monkeypatch.setenv("LOCAL_RANK", "1")
	monkeypatch.setenv("PL_GLOBAL_SEED", "1234")

	captured = _run_capturing_training_fit(fit_cfg)
	assert captured['peak_generator']['random_state'] == 1234


def test_fit_first_rank_ignores_a_seed_left_in_the_environment(fit_cfg,
	monkeypatch, capsys):
	fit_cfg.random_state = None

	monkeypatch.delenv("LOCAL_RANK", raising=False)
	monkeypatch.setenv("PL_GLOBAL_SEED", "1234")

	captured = _run_capturing_training_fit(fit_cfg)
	drawn = captured['peak_generator']['random_state']
	assert "Drew random_state={}".format(drawn) in capsys.readouterr().out


def _srun_seed(monkeypatch, procid, step):
	monkeypatch.delenv("LOCAL_RANK", raising=False)
	monkeypatch.setenv("SLURM_NTASKS", "2")
	monkeypatch.setenv("SLURM_JOB_ID", "4242")
	monkeypatch.setenv("SLURM_STEP_ID", str(step))
	monkeypatch.setenv("SLURM_PROCID", str(procid))

	captured = _run_capturing_training_fit(_fit_config(random_state=None))
	return captured['peak_generator']['random_state']


def test_fit_srun_ranks_derive_the_same_seed(monkeypatch, capsys):
	"""`srun` starts every rank at once, so no rank can inherit another's
	draw. Each derives the seed from the job step, so they agree, and a new
	`srun` gets a new seed."""

	first = _srun_seed(monkeypatch, procid=0, step=0)
	assert "Derived random_state={}".format(first) in capsys.readouterr().out

	assert _srun_seed(monkeypatch, procid=1, step=0) == first
	assert _srun_seed(monkeypatch, procid=0, step=1) != first


def _fit_and_pipeline_defaults():
	"""The composed `fit` config and the fit node of the `pipeline` config,
	with every required key set."""

	from cherimoya_cli.config import PipelineConfig

	pipeline = OmegaConf.structured(PipelineConfig(name='demo',
		sequences='fake.fa', loci='fake.bed', negatives='fake_negatives.bed',
		signals=['fake.bw']))
	return _fit_config(), pipeline.fit


def test_default_num_workers_is_one():
	for cfg in _fit_and_pipeline_defaults():
		assert cfg.num_workers == 1


def test_default_disables_early_stopping():
	"""Early stopping is off by default in both the fit config and the
	fit node of the pipeline config, matching `cherimoya.training.fit`'s
	``early_stopping=None``. A non-None default here would cut the cosine
	learning rate schedule -- laid out over ``max_epochs`` -- short."""

	for cfg in _fit_and_pipeline_defaults():
		assert cfg.early_stopping is None


def test_default_progress_bar_is_none():
	for cfg in _fit_and_pipeline_defaults():
		assert cfg.progress_bar is None


def test_fit_forwards_grouped_signals_to_peak_generator():
	"""When the config gives a grouped signals spec, fit.run must
	forward the inferred signal_groups (and control_groups) to
	PeakGenerator. Without this the structured form would be flattened
	to "all unstranded" downstream."""

	from cherimoya_cli.commands import fit as fit_cmd

	# One unstranded ATAC group + one stranded TF group; per-group
	# counts mean signal_groups=[1, 2].
	cfg = _fit_config(signals=['atac.bw', ['ctcf.+.bw', 'ctcf.-.bw']],
		controls=[['ctl.+.bw', 'ctl.-.bw']], name='fit_wiring_groups_test')

	captured = {}

	class _StopFit(Exception):
		pass

	def fake_peak_generator(**kwargs):
		captured.update(kwargs)
		raise _StopFit()

	with mock.patch(
			"cherimoya.io.PeakGenerator", side_effect=fake_peak_generator):
		try:
			fit_cmd.run(cfg)
		except _StopFit:
			pass
		except Exception:
			if not captured:
				raise

	assert captured.get('signal_groups') == [1, 2], (
		"signal_groups not forwarded; got {!r}".format(
			captured.get('signal_groups')))
	assert captured.get('control_groups') == [2], (
		"control_groups not forwarded; got {!r}".format(
			captured.get('control_groups')))


def test_fit_does_not_flatten_signals_for_downstream_evaluate(tmp_path):
	"""fit.run hands ``parameters['signals']`` to PeakGenerator and
	(at the end of training) copies the same value into the
	evaluate configs. If fit silently re-writes ``signals`` to its flat
	form, a stranded pair ``[[+, -]]`` becomes ``[+, -]`` in the
	evaluate configs, which then re-parse as two *unstranded* channels
	— the same bug the grouping API was added to prevent. Pin the
	contract by snapshotting what PeakGenerator actually receives."""

	from cherimoya_cli.commands import fit as fit_cmd

	original_signals = [['ctcf.+.bw', 'ctcf.-.bw']]
	cfg = _fit_config(signals=original_signals,
		name=str(tmp_path / 'fit_eval_roundtrip'))

	captured = {}

	class _StopFit(Exception):
		pass

	def fake_peak_generator(**kwargs):
		captured.update(kwargs)
		raise _StopFit()

	with mock.patch(
			"cherimoya.io.PeakGenerator", side_effect=fake_peak_generator):
		try:
			fit_cmd.run(cfg)
		except _StopFit:
			pass
		except Exception:
			if not captured:
				raise

	# PeakGenerator must see the *structured* signals form. If fit had
	# pre-flattened it the assertion below would fail with
	# `signals == ['ctcf.+.bw', 'ctcf.-.bw']` (two unstranded tracks).
	assert captured.get('signals') == original_signals, (
		"fit flattened the structured signals form before PeakGenerator: "
		"got {!r}".format(captured.get('signals')))


def _run_capturing_peak_generator(cfg):
	"""Run fit.run and return the kwargs PeakGenerator was called with.

	Execution stops at that call, which is far enough to see every
	parameter the sampler is given but short of any real IO.
	"""

	from cherimoya_cli.commands import fit as fit_cmd

	captured = {}

	class _StopFit(Exception):
		pass

	def fake_peak_generator(**kwargs):
		captured.update(kwargs)
		raise _StopFit()

	with mock.patch("cherimoya.io.PeakGenerator",
			side_effect=fake_peak_generator):
		try:
			fit_cmd.run(cfg)
		except _StopFit:
			pass
		except Exception:
			if not captured:
				raise

	assert captured, "PeakGenerator was never called"
	return captured


def test_default_random_state_is_zero():
	"""Training is seeded by default. The pipeline's fit node
	interpolates the top-level seed rather than holding its own, so a
	value there cannot shadow whatever the user set at the top level."""

	fit, pipeline_fit = _fit_and_pipeline_defaults()
	assert fit.random_state == 0
	assert pipeline_fit.random_state == 0
	assert OmegaConf.to_container(pipeline_fit)['random_state'] == (
		'${random_state}')


def test_pipeline_random_state_reaches_the_fit_node():
	"""The top-level pipeline seed must reach the fit node, since that is
	the config the fit step actually reads."""

	from cherimoya_cli.config import PipelineConfig

	cfg = OmegaConf.structured(PipelineConfig)
	cfg.random_state = 7
	assert cfg.fit.random_state == 7


def test_fit_forwards_random_state_to_peak_generator(fit_cfg):
	"""The sampler's draw order is the half of reproducibility that was
	already wired; confirm an explicit seed still reaches it."""

	fit_cfg.random_state = 42

	captured = _run_capturing_peak_generator(fit_cfg)
	assert captured.get('random_state') == 42


def test_fit_seeds_with_the_default_random_state(fit_cfg):
	"""A config that leaves the seed out falls back to 0 rather than
	running unseeded."""

	captured = _run_capturing_peak_generator(fit_cfg)
	assert captured.get('random_state') == 0


def test_fit_draws_and_announces_a_null_random_state(fit_cfg, capsys):
	"""A null seed means "pick one and tell me", not "stay unseeded".
	The drawn value has to be printed regardless of `verbose`, because
	it is recorded nowhere else. It is also written back into the
	config the caller passed."""

	fit_cfg.random_state = None
	fit_cfg.verbose = False

	captured = _run_capturing_peak_generator(fit_cfg)

	drawn = captured.get('random_state')
	assert isinstance(drawn, int), (
		"a null random_state must be resolved to an integer before the "
		"sampler is built; got {!r}".format(drawn))
	assert fit_cfg.random_state == drawn

	out = capsys.readouterr().out
	assert "Drew random_state={}".format(drawn) in out


def test_fit_seeds_the_model_initialization(fit_cfg):
	"""The seed has to reach the model as well as the sampler — the
	initialization is the larger source of run-to-run variance, and it
	was the half that nothing seeded before."""

	import torch

	from cherimoya_cli.commands import fit as fit_cmd

	captured = {}

	class _StopFit(Exception):
		pass

	def fake_extract_loci(**kwargs):
		# (sequences, signals); controls are absent in the fixture config.
		return torch.zeros(1, 4, 16), torch.zeros(1, 1, 8)

	def fake_model(**kwargs):
		captured.update(kwargs)
		raise _StopFit()

	with mock.patch("cherimoya.io.PeakGenerator",
				return_value=_FakeLoader([None] * 4)), \
			mock.patch("tangermeme.io._interleave_loci", return_value=[None]), \
			mock.patch("tangermeme.io.extract_loci",
				side_effect=fake_extract_loci), \
			mock.patch("cherimoya.Cherimoya", side_effect=fake_model):
		try:
			fit_cmd.run(fit_cfg)
		except _StopFit:
			pass
		except Exception:
			if not captured:
				raise

	assert captured, "Cherimoya was never constructed"
	assert captured.get('random_state') == 0


# --------- the minimum step count ----------------------------------------
#
# These import `_max_epochs_for_min_steps` from the fit command rather than
# restating the rule, so an edit to it is caught here instead of silently
# diverging from a copy.

def test_default_min_total_steps_is_twenty_thousand():
	"""An epoch is one pass over the peaks, so `max_epochs` alone buys a
	step count proportional to peak count. The floor is on by default in
	both the fit config and the fit node of the pipeline config."""

	for cfg in _fit_and_pipeline_defaults():
		assert cfg.min_total_steps == 20000


def test_min_steps_extends_a_short_run():
	"""14 batches of peaks over 20 epochs is 280 steps, far under the
	floor, so the run is extended to the epoch count that reaches it."""

	from cherimoya_cli.commands.fit import _max_epochs_for_min_steps

	assert _max_epochs_for_min_steps(20, 14, 20000) == 1429
	assert 1429 * 14 >= 20000


def test_min_steps_rounds_up_rather_than_landing_short():
	"""The division has to round up: 20 epochs of 999 is 19,980, and an
	epoch count that lands one step under the floor would defeat it."""

	from cherimoya_cli.commands.fit import _max_epochs_for_min_steps

	assert _max_epochs_for_min_steps(20, 999, 20000) == 21
	assert 21 * 999 >= 20000


def test_min_steps_leaves_a_long_enough_run_alone():
	"""An accessibility-shaped experiment has thousands of batches per
	epoch and already clears the floor, so nothing changes. Equality
	counts as clearing it."""

	from cherimoya_cli.commands.fit import _max_epochs_for_min_steps

	assert _max_epochs_for_min_steps(20, 2700, 20000) == 20
	assert _max_epochs_for_min_steps(20, 1000, 20000) == 20


def test_min_steps_null_disables_the_floor():
	"""`min_total_steps: null` is the escape hatch for a short run, e.g. a
	smoke test that sets max_epochs to 1."""

	from cherimoya_cli.commands.fit import _max_epochs_for_min_steps

	assert _max_epochs_for_min_steps(20, 14, None) == 20
	assert _max_epochs_for_min_steps(1, 14, None) == 1


def test_min_steps_tolerates_an_empty_training_set():
	"""`len(training_data)` can be zero if every locus was filtered out.
	Dividing by it would raise before the clearer error downstream."""

	from cherimoya_cli.commands.fit import _max_epochs_for_min_steps

	assert _max_epochs_for_min_steps(20, 0, 20000) == 20


# --------- fixed loss weights ---------------------------------------------

def test_default_loss_weights_is_none():
	"""The Kendall weights stay the default. Turning the fixed weights on
	is opt-in, in both the fit config and the fit node of the pipeline
	config."""

	for cfg in _fit_and_pipeline_defaults():
		assert cfg.loss_weights is None


def test_loss_weights_override_survives_as_a_pair(make_config):
	"""An override gives the pair as a list; ``fit`` unpacks it either way
	and the values must not be coerced or reordered."""

	cfg = make_config("fit", "loss_weights=[1.333,0.274]")
	w0, w1 = cfg.loss_weights
	assert (w0, w1) == (1.333, 0.274)


# --------- the verbose loss-balance line ----------------------------------
#
# `_loss_balance_summary` is tested directly, where the rule lives; the
# banner test below checks that `fit.run` prints its line.

def test_loss_balance_summary_reports_the_sgd_optimizer_by_default():
	"""With the Kendall weights in use, `lw_optimizer` is training `lw0`
	and `lw1`, so its hyperparameters are what the run is governed by."""

	from cherimoya_cli.commands.fit import _loss_balance_summary

	summary = _loss_balance_summary(None, 0.01, 0.0, 0.9)
	assert summary == "SGD Optimizer (lw): lr=0.01, wd=0.0, momentum=0.9"


def test_loss_balance_summary_reports_the_fixed_weights():
	"""When `loss_weights` is given, `lw0` and `lw1` stop receiving
	gradient, so the SGD hyperparameters describe an optimizer that does
	nothing. The constants actually balancing the two losses are reported
	instead, and none of the SGD values leak into the line."""

	from cherimoya_cli.commands.fit import _loss_balance_summary

	summary = _loss_balance_summary((1.333, 0.274), 0.01, 0.0, 0.9)
	assert "1.333" in summary
	assert "0.274" in summary
	assert "SGD" not in summary
	assert "momentum" not in summary


def test_loss_balance_summary_accepts_the_pair_as_a_list():
	"""A config has no tuple, so `loss_weights` arrives as a list and must
	be reported the same way a tuple is."""

	from cherimoya_cli.commands.fit import _loss_balance_summary

	pair = OmegaConf.create([1.333, 0.274])
	assert (_loss_balance_summary(pair, 0.01, 0.0, 0.9)
		== _loss_balance_summary((1.333, 0.274), 0.01, 0.0, 0.9))


@pytest.mark.parametrize("loss_weights,expected,forbidden", [
	(None, "SGD Optimizer (lw): lr=", "Fixed Loss Weights"),
	([1.333, 0.274], "Fixed Loss Weights: profile=1.333, count=0.274",
		"SGD Optimizer (lw)"),
])
def test_fit_banner_names_the_loss_balancing_in_force(fit_cfg, capsys,
	loss_weights, expected, forbidden):
	"""End to end through `fit.run`: with `verbose` on, the banner must
	name whichever scheme the run uses -- the `lw_*` optimizer when the
	weights are learned, the constants when they are fixed and that
	optimizer is inert. Training is cut off at `cherimoya.training.fit`,
	which is the call the banner immediately precedes."""

	import torch

	from cherimoya_cli.commands import fit as fit_cmd

	fit_cfg.loss_weights = loss_weights
	fit_cfg.verbose = True
	fit_cfg.n_layers = 2
	fit_cfg.n_filters = 8

	class _StopFit(Exception):
		pass

	def fake_extract_loci(**kwargs):
		return torch.zeros(1, 4, 16), torch.zeros(1, 1, 8)

	with mock.patch("cherimoya.io.PeakGenerator",
				return_value=_FakeLoader([None] * 4)), \
			mock.patch("tangermeme.io._interleave_loci", return_value=[None]), \
			mock.patch("tangermeme.io.extract_loci",
				side_effect=fake_extract_loci), \
			mock.patch("cherimoya.training.fit", side_effect=_StopFit()):
		with pytest.raises(_StopFit):
			fit_cmd.run(fit_cfg)

	out = capsys.readouterr().out
	assert expected in out
	assert forbidden not in out


# --------- the evaluate step ----------------------------------------------

def _run_fit_evaluations(cfg, tmp_path, monkeypatch, train=None):
	"""Run `fit.run` through its evaluate step with training and data
	loading faked, and return the evaluate configs it passed on, in order.
	`train`, if given, is called in place of training and returns
	`mock.DEFAULT` for the faked trainer."""

	import types

	import torch

	from cherimoya_cli.commands import fit as fit_cmd

	monkeypatch.chdir(tmp_path)

	def fake_extract_loci(**kwargs):
		return torch.zeros(1, 4, 16), torch.zeros(1, 1, 8)

	with mock.patch("cherimoya.io.PeakGenerator",
				return_value=_FakeLoader([None] * 4)), \
			mock.patch("tangermeme.io._interleave_loci", return_value=[None]), \
			mock.patch("tangermeme.io.extract_loci",
				side_effect=fake_extract_loci), \
			mock.patch("cherimoya.training.fit", side_effect=train,
				return_value=types.SimpleNamespace(is_global_zero=True)), \
			mock.patch("cherimoya_cli.commands.evaluate.run") as evaluate:
		fit_cmd.run(cfg)

	return [call.args[0] for call in evaluate.call_args_list]


def test_fit_leaves_no_ranks_behind_for_the_next_job(fit_cfg, tmp_path,
	monkeypatch):
	"""A sweep runs its jobs one after another in one process. Lightning
	keeps a multi-device job's process group and the rank variables it
	set, so the next job would start no ranks of its own and reach this
	job's exited ones instead."""

	import os

	import torch.distributed as dist

	monkeypatch.delenv("LOCAL_RANK", raising=False)
	entered = []

	def train(*args, **kwargs):
		entered.append(("LOCAL_RANK" in os.environ, dist.is_initialized()))
		os.environ.update(LOCAL_RANK="0", WORLD_SIZE="1")
		# Lightning keeps a process group that is already there.
		if not dist.is_initialized():
			dist.init_process_group("gloo", rank=0, world_size=1,
				init_method=(tmp_path / str(len(entered))).as_uri())
		return mock.DEFAULT

	with mock.patch.dict(os.environ):
		try:
			for _ in range(2):
				_run_fit_evaluations(fit_cfg, tmp_path, monkeypatch, train)
		finally:
			if dist.is_initialized():
				dist.destroy_process_group()

	assert entered == [(False, False), (False, False)]


def test_fit_evaluates_the_validation_and_test_chromosomes(fit_cfg, tmp_path,
	monkeypatch):
	"""Each evaluate config is saved next to the model, so either
	evaluation can be rerun with `-p`."""

	from cherimoya_cli.config import test_chroms, validation_chroms

	evaluated = _run_fit_evaluations(fit_cfg, tmp_path, monkeypatch)

	assert [(list(cfg.chroms), cfg.performance_filename, cfg.model)
		for cfg in evaluated] == [
		(validation_chroms, "fit_wiring_test.validation.performance.tsv",
			"fit_wiring_test.torch"),
		(test_chroms, "fit_wiring_test.test.performance.tsv",
			"fit_wiring_test.torch"),
	]
	for split, cfg in zip(("validation", "test"), evaluated):
		saved = OmegaConf.load(tmp_path / "fit_wiring_test.{}.evaluate.yaml"
			.format(split))
		assert saved == cfg


def test_fit_skips_the_test_evaluation_without_test_chromosomes(fit_cfg,
	tmp_path, monkeypatch):
	fit_cfg.test_chroms = None

	evaluated = _run_fit_evaluations(fit_cfg, tmp_path, monkeypatch)
	assert [cfg.performance_filename for cfg in evaluated] == [
		"fit_wiring_test.validation.performance.tsv"]
	assert not (tmp_path / "fit_wiring_test.test.evaluate.yaml").exists()


def test_fit_hands_evaluate_only_the_keys_evaluate_declares(fit_cfg,
		tmp_path, monkeypatch):
	"""Evaluate's schema rejects unknown keys, so fit copies only the
	shared ones, the negatives and summits among them, and points the rest
	at the trained model."""

	import dataclasses

	from cherimoya_cli.config import EvaluateConfig

	fit_cfg.signals = [['ctcf.+.bw', 'ctcf.-.bw']]
	fit_cfg.summits = True
	cfg, _ = _run_fit_evaluations(fit_cfg, tmp_path, monkeypatch)

	assert OmegaConf.get_type(cfg) is EvaluateConfig
	assert set(cfg) == {f.name for f in dataclasses.fields(EvaluateConfig)}
	assert cfg.batch_size == 16
	assert cfg.negatives == "fake_negatives.bed"
	assert cfg.summits is True
	assert OmegaConf.to_container(cfg.signals) == [['ctcf.+.bw', 'ctcf.-.bw']]


@pytest.mark.parametrize("key", ["validation_chroms", "test_chroms"])
def test_fit_refuses_chromosomes_shared_between_splits(fit_cfg, key):
	from cherimoya_cli.commands import fit as fit_cmd

	fit_cfg[key] = list(fit_cfg[key]) + ["chr2"]

	with pytest.raises(ValueError, match="training_chroms and {} share "
		r"\['chr2'\]".format(key)):
		fit_cmd.run(fit_cfg)


def test_fit_builds_the_model_with_the_compile_setting(fit_cfg):
	fit_cfg.compile = False
	fit_cfg.n_filters, fit_cfg.n_layers = 8, 2

	model = _run_capturing_training_fit(fit_cfg)['model']
	assert model._compile is False


def test_fit_centers_validation_peaks_as_training_does(fit_cfg):
	"""With `summits`, the validation peaks are centered on the summit like
	the training peaks; the negatives, which have no summit, are not."""

	fit_cfg.summits = True

	captured = _run_capturing_training_fit(fit_cfg)
	assert captured['peak_generator']['summits'] is True
	assert captured['extract']['fake.bed']['summits'] is True
	assert captured['extract']['fake_negatives.bed'].get('summits', False) is False


def test_fit_validates_without_negatives_on_the_validation_chromosomes(
	fit_cfg):
	"""A negatives file with nothing on the validation chromosomes leaves
	the validation set to the peaks rather than failing in extract_loci."""

	captured = _run_capturing_training_fit(fit_cfg, n_on_chroms=0)
	X_valid, _ = captured['valid']
	assert len(X_valid) == 1
	assert captured['labels_valid'] is None
	assert 'fake_negatives.bed' not in captured['extract']
