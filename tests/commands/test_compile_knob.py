"""Tests that the inference subcommands expose `torch.compile` settings.

`Cherimoya.load` compiles the forward with ``mode='max-autotune'`` by
default, and the troubleshooting page and the bundled skill both tell
users to pass ``compile=False`` when they hit a CUDA-graph error or want
to attribute a model. Without a JSON key for it there was no way to act
on that advice from the CLI.
"""

import dataclasses
from unittest import mock

import pytest
from omegaconf import OmegaConf

from cherimoya_cli.config import AttributeConfig
from cherimoya_cli.config import EvaluateConfig
from cherimoya_cli.config import MarginalizeConfig


# Every subcommand that loads a model, and so has to carry both knobs.
ALL_SCHEMAS = {
	"attribute": AttributeConfig,
	"evaluate": EvaluateConfig,
	"marginalize": MarginalizeConfig,
}

# A value for every required key of any of them.
REQUIRED = {"model": "m.torch", "sequences": "g.fa", "loci": "x.bed",
	"signals": ["s.bw"], "motifs": "m.meme"}


def _captured_load_kwargs(command, tmp_path, **overrides):
	"""Run a subcommand far enough to capture the kwargs it passes to
	`Cherimoya.load`, then abort.

	Every required key is given a value, so `Cherimoya.load` -- the
	first thing each of these commands does after the `skip` guard -- is
	reached without needing real inputs.
	"""

	captured = {}

	class _Stop(Exception):
		pass

	def fake_load(path, **kwargs):
		captured.update(kwargs)
		raise _Stop()

	schema = ALL_SCHEMAS[command]
	values = {field.name: REQUIRED[field.name]
		for field in dataclasses.fields(schema) if field.name in REQUIRED}
	# `attribute` loads uncompiled for DeepLIFT whatever `compile` says
	# (see test_attribute.py); the knob applies to its ISM path.
	if command == "attribute":
		values["algorithm"] = "saturation_mutagenesis"
	cfg = OmegaConf.structured(schema(**values, **overrides))

	mod = __import__("cherimoya_cli.commands." + command, fromlist=["run"])

	with mock.patch("cherimoya.Cherimoya") as model_cls:
		model_cls.load.side_effect = fake_load
		with pytest.raises(_Stop):
			mod.run(cfg)

	return captured


##


@pytest.mark.parametrize("command", ["evaluate", "marginalize"])
def test_default_is_compiled(command, tmp_path):
	"""Omitting the keys preserves the pre-existing behaviour: the same
	settings `Cherimoya.load` uses on its own."""

	captured = _captured_load_kwargs(command, tmp_path)

	assert captured["compile"] is True
	assert captured["compile_mode"] == "max-autotune"


def test_attribute_default_is_uncompiled(tmp_path):
	"""`attribute` defaults to an eager model: neither algorithm ran
	faster compiled, and compiling only lengthened the first call."""

	captured = _captured_load_kwargs("attribute", tmp_path)

	assert captured["compile"] is False


@pytest.mark.parametrize("command", sorted(ALL_SCHEMAS))
@pytest.mark.parametrize("compile_flag", [True, False])
def test_forwards_compile(command, compile_flag, tmp_path):
	"""The value in the config is what reaches `Cherimoya.load`."""

	captured = _captured_load_kwargs(command, tmp_path,
		compile=compile_flag)

	assert captured["compile"] is compile_flag


@pytest.mark.parametrize("command", sorted(ALL_SCHEMAS))
def test_forwards_compile_mode(command, tmp_path):
	captured = _captured_load_kwargs(command, tmp_path,
		compile_mode="max-autotune-no-cudagraphs")

	assert captured["compile_mode"] == "max-autotune-no-cudagraphs"


def test_pipeline_shares_the_compile_setting(tmp_path, run_pipeline):
	"""Set once at the pipeline top level, the value reaches the
	per-step YAMLs, the same way `dtype` and `device` do."""

	from cherimoya_cli.config import PipelineConfig

	assert PipelineConfig.compile is True
	assert PipelineConfig.compile_mode == "max-autotune"

	run_pipeline(compile=False, compile_mode="reduce-overhead",
		motifs=str(tmp_path / "m.meme"))

	emitted = OmegaConf.load(tmp_path / "demo.marginalize.yaml")
	assert emitted.compile is False
	assert emitted.compile_mode == "reduce-overhead"

	# The attribute step pins its own `compile` but inherits the mode.
	emitted = OmegaConf.load(tmp_path / "demo.attribute.yaml")
	assert emitted.compile_mode == "reduce-overhead"

	# fit builds the training model with the setting and copies it into
	# both evaluate configs.
	emitted = OmegaConf.load(tmp_path / "demo.fit.yaml")
	assert emitted.compile is False
	assert emitted.compile_mode == "reduce-overhead"
