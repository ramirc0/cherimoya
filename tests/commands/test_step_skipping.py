"""Tests that a skipped step returns instead of killing the process.

`cherimoya pipeline` calls each subcommand's `run(args)` in-process, so
a `sys.exit()` inside one of them ends the whole pipeline rather than
the step. `"skip": true` is documented as no-opping a step, and the
marginalization guard is documented as skipping a stage when no motif
database is given; both terminated the interpreter instead.
"""

import pytest


##


@pytest.mark.parametrize("command", ["fit", "evaluate", "attribute",
	"seqlets", "marginalize"])
def test_skip_returns_rather_than_exiting(command):
	"""Every subcommand that honours `skip` must return, so the caller
	decides what happens next."""

	import dataclasses

	from omegaconf import MISSING, OmegaConf

	from cherimoya_cli.config import SCHEMAS

	schema = SCHEMAS[command]
	required = {f.name: "x" for f in dataclasses.fields(schema)
		if f.default == MISSING}

	mod = __import__("cherimoya_cli.commands." + command, fromlist=["run"])
	cfg = OmegaConf.structured(schema(skip=True, **required))

	# Returns None rather than raising SystemExit.
	assert mod.run(cfg) is None


def test_pipeline_without_motifs_returns(run_pipeline):
	"""The marginalization guard skips a stage, not the interpreter."""

	assert run_pipeline(motifs=None) is None


def test_pipeline_skip_runs_no_step(tmp_path, run_pipeline):
	"""A top-level `skip` ends the pipeline before any step, including the
	preprocessing and MoDISco commands that have no `skip` of their own.
	No step config is saved either."""

	from unittest import mock

	with mock.patch("subprocess.run") as subprocess_run:
		run_pipeline(skip=True, dry_run=False, loci=None)

	subprocess_run.assert_not_called()
	assert not list(tmp_path.glob("demo.*.yaml"))


def test_pipeline_skips_annotation_when_asked(tmp_path, pipeline_config,
		monkeypatch):
	from unittest import mock

	from cherimoya_cli.commands import pipeline

	cfg = pipeline_config(motifs=str(tmp_path / "m.meme"), dry_run=False)
	cfg.annotation.skip = True
	monkeypatch.chdir(tmp_path)

	with mock.patch("subprocess.run") as subprocess_run, \
			mock.patch("cherimoya_cli.commands.fit.run"), \
			mock.patch("cherimoya_cli.commands.attribute.run"), \
			mock.patch("cherimoya_cli.commands.seqlets.run"), \
			mock.patch("cherimoya_cli.commands.marginalize.run"):
		pipeline.run(cfg)

	commands = [call.args[0][0] for call in subprocess_run.call_args_list]
	assert "ttl" not in commands
	assert "modisco" in commands
