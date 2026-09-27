"""Tests for the keys a pipeline config is allowed to leave out.

A user's `run.yaml` names only what it changes, so every key the
pipeline reads unguarded must still have a value when the file omits it.
"""

import sys

import pytest


# Each has a default, so a config that omits it must come out of
# composition with it set rather than leaving `pipeline.run` to trip over
# a missing key.
OPTIONAL_KEYS = ("motifs", "model", "controls")


##


@pytest.mark.parametrize("key", OPTIONAL_KEYS)
def test_pipeline_config_may_omit_an_optional_key(key, pipeline_config):
	assert pipeline_config()[key] is None


def test_pipeline_dry_run_without_the_optional_keys(run_pipeline):
	"""The end-to-end symptom: a config that omits all three must get
	through the run rather than dying on a missing key.

	The run stops at the marginalization step because `motifs` is None,
	which is `pipeline.run`'s own control flow and not an error, so it
	returns rather than raising.
	"""

	assert run_pipeline() is None


def test_missing_key_error_names_the_null_fix(tmp_path, monkeypatch,
		capsys):
	"""A key that really is required still fails, and the message says
	what to write: `null` is accepted and a missing key is not, which is
	not guessable from 'Must provide a value'."""

	from cherimoya_cli.__main__ import main

	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr(sys, "argv", ["cherimoya", "pipeline", "name=demo",
		"sequences=g.fa", "signals=[s.bw]"])

	with pytest.raises(SystemExit):
		main()

	err = capsys.readouterr().err
	assert "Must provide a value for: loci, negatives." in err
	assert "null" in err


def test_null_random_state_is_drawn_once_for_every_step(run_pipeline,
		tmp_path, capsys):
	"""A null seed means "draw one and print it". The pipeline draws it
	before any step, so fit, attribute and marginalize share it. It used
	to be drawn inside fit, which left attribute and marginalize
	unseeded."""

	from omegaconf import OmegaConf

	run_pipeline(random_state=None, motifs=str(tmp_path / "m.meme"))

	seeds = {step: OmegaConf.load(tmp_path / "demo.{}.yaml".format(step))
		.random_state for step in ("fit", "attribute", "marginalize")}
	drawn = seeds["fit"]
	assert isinstance(drawn, int)
	assert seeds == dict.fromkeys(seeds, drawn)
	assert "Drew random_state={}".format(drawn) in capsys.readouterr().out


def test_a_multi_device_fit_runs_as_its_own_command(tmp_path,
		pipeline_config, monkeypatch):
	"""Lightning starts each rank after the first by re-running the current
	command, which in-process would repeat the whole pipeline. With more
	than one device the fit runs as its own command on the saved fit
	config instead."""

	from unittest import mock

	from omegaconf import OmegaConf

	from cherimoya_cli.commands import pipeline

	cfg = pipeline_config(dry_run=False)
	cfg.fit.devices = 2
	monkeypatch.chdir(tmp_path)

	with mock.patch("subprocess.run") as subprocess_run, \
			mock.patch("cherimoya_cli.commands.fit.run") as fit_run, \
			mock.patch("cherimoya_cli.commands.attribute.run"), \
			mock.patch("cherimoya_cli.commands.seqlets.run"):
		pipeline.run(cfg)

	fit_run.assert_not_called()
	assert mock.call([sys.executable, "-m", "cherimoya_cli", "fit", "-p",
		"demo.fit.yaml"], check=True) in subprocess_run.call_args_list
	assert OmegaConf.load(tmp_path / "demo.fit.yaml").devices == 2
