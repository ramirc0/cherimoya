"""Shared fixtures for the `cherimoya_cli` subcommand tests."""

import pytest


# Written as empty files so the configs name inputs that exist.
PIPELINE_INPUTS = ("g.fa", "x.bed", "n.bed", "s.bw", "m.meme")


@pytest.fixture
def pipeline_config(tmp_path, make_config):
	"""Factory composing a `pipeline` config that names real input files,
	with `dry_run` on.

	Keyword arguments set config values after composing, the way the
	pipeline itself assigns the keys it produces.
	"""

	def build(**overrides):
		for name in PIPELINE_INPUTS:
			(tmp_path / name).write_text("")

		cfg = make_config("pipeline", "name=demo",
			"sequences={}".format(tmp_path / "g.fa"),
			"loci=[{}]".format(tmp_path / "x.bed"),
			"negatives=[{}]".format(tmp_path / "n.bed"),
			"signals=[{}]".format(tmp_path / "s.bw"),
			"dry_run=true", "verbose=false")
		for key, value in overrides.items():
			cfg[key] = value
		return cfg

	return build


@pytest.fixture
def run_pipeline(pipeline_config, tmp_path, monkeypatch):
	"""Run `cherimoya pipeline` from `tmp_path` on a config built by
	`pipeline_config`.

	The working directory matters: `dry_run` still writes each step's
	YAML, and it writes them relative to the working directory.
	"""

	def run(**overrides):
		from cherimoya_cli.commands import pipeline

		cfg = pipeline_config(**overrides)
		monkeypatch.chdir(tmp_path)
		return pipeline.run(cfg)

	return run
