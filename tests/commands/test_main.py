"""Tests for the `cherimoya` entry point.

The Hydra commands are dispatched before argparse sees the arguments:
`-p FILE` is layered over the command's schema, `key=value` overrides
apply on top, and the composed config is checked for missing keys and
missing input files before the command runs.
"""

import sys
from unittest import mock

import pytest


SEQLETS_INPUTS = ("x.bed", "a.ohe.npz", "a.attr.npz", "a.idx.npy")


def _main(monkeypatch, *argv):
	from cherimoya_cli.__main__ import main

	monkeypatch.setattr(sys, "argv", ["cherimoya", *argv])
	main()


@pytest.fixture
def seqlets_inputs(tmp_path, monkeypatch):
	"""Empty files for every input `seqlets` reads, in the working
	directory, and a patched command that records the config it gets."""

	for name in SEQLETS_INPUTS:
		(tmp_path / name).write_text("")
	monkeypatch.chdir(tmp_path)

	with mock.patch("cherimoya_cli.commands.seqlets.run") as run:
		yield run


def _yaml(tmp_path, text):
	path = tmp_path / "run.yaml"
	path.write_text(text)
	return str(path)


##


def test_file_and_overrides_compose(tmp_path, monkeypatch, seqlets_inputs):
	"""The file sets the keys it names, an override beats the file, and
	everything else keeps the schema default."""

	path = _yaml(tmp_path, "loci: x.bed\nohe_filename: a.ohe.npz\n"
		"attr_filename: a.attr.npz\nidx_filename: a.idx.npy\n"
		"threshold: 0.2\nmin_seqlet_len: 6\n")

	_main(monkeypatch, "seqlets", "-p", path, "threshold=0.5")

	cfg = seqlets_inputs.call_args.args[0]
	assert cfg.threshold == 0.5
	assert cfg.min_seqlet_len == 6
	assert cfg.max_seqlet_len == 25


@pytest.mark.parametrize("flag", ["-m", "--cfg job"])
def test_a_flag_after_the_overrides_keeps_the_file(tmp_path, monkeypatch,
		capsys, seqlets_inputs, flag):
	"""Hydra takes one unbroken run of overrides, so a flag typed after
	them must not split the file's layer from the rest."""

	path = _yaml(tmp_path, "loci: x.bed\nohe_filename: a.ohe.npz\n"
		"attr_filename: a.attr.npz\nidx_filename: a.idx.npy\n"
		"min_seqlet_len: 6\n")

	_main(monkeypatch, "seqlets", "-p", path, "threshold=0.5", *flag.split())

	if flag == "-m":
		cfg = seqlets_inputs.call_args.args[0]
	else:
		from omegaconf import OmegaConf

		cfg = OmegaConf.create(capsys.readouterr().out)
	assert (cfg.threshold, cfg.min_seqlet_len) == (0.5, 6)


def test_inputs_are_made_absolute(tmp_path, monkeypatch, seqlets_inputs):
	_main(monkeypatch, "seqlets", "loci=x.bed", "ohe_filename=a.ohe.npz",
		"attr_filename=a.attr.npz", "idx_filename=a.idx.npy")

	cfg = seqlets_inputs.call_args.args[0]
	assert cfg.loci == str(tmp_path / "x.bed")


@pytest.mark.parametrize("where", ["file", "override"])
def test_a_typo_is_rejected(tmp_path, monkeypatch, capsys, seqlets_inputs,
		where):
	argv = ["seqlets", "loci=x.bed"]
	if where == "file":
		argv += ["-p", _yaml(tmp_path, "threshld: 0.5\n")]
	else:
		argv += ["threshld=0.5"]

	with pytest.raises(SystemExit):
		_main(monkeypatch, *argv)

	assert "threshld" in capsys.readouterr().err
	seqlets_inputs.assert_not_called()


def test_every_missing_key_is_listed(monkeypatch, capsys, seqlets_inputs):
	with pytest.raises(SystemExit):
		_main(monkeypatch, "seqlets")

	err = capsys.readouterr().err
	assert ("Must provide a value for: attr_filename, idx_filename, loci, "
		"ohe_filename") in err
	seqlets_inputs.assert_not_called()


def test_every_missing_input_is_listed(tmp_path, monkeypatch, capsys,
		seqlets_inputs):
	with pytest.raises(SystemExit):
		_main(monkeypatch, "seqlets", "loci=[y.bed,z.bed]",
			"ohe_filename=a.ohe.npz", "attr_filename=a.attr.npz",
			"idx_filename=a.idx.npy")

	err = capsys.readouterr().err
	assert "loci: {}".format(tmp_path / "y.bed") in err
	assert "loci: {}".format(tmp_path / "z.bed") in err
	seqlets_inputs.assert_not_called()


def test_the_command_receives_its_own_config(tmp_path, monkeypatch):
	"""Each Hydra command is routed to its own module and schema."""

	for name in ("p.bed", "g.fa"):
		(tmp_path / name).write_text("")
	monkeypatch.chdir(tmp_path)

	with mock.patch("cherimoya_cli.commands.negatives.run") as run:
		_main(monkeypatch, "negatives", "peaks=p.bed", "fasta=g.fa",
			"output=n.bed", "beta=0.7")

	cfg = run.call_args.args[0]
	assert cfg.peaks == str(tmp_path / "p.bed")
	assert cfg.output == "n.bed"
	assert cfg.beta == 0.7


def test_fit_is_dispatched_to_hydra(tmp_path, monkeypatch):
	"""Grouped signals keep their nesting, and every path in them is made
	absolute."""

	for name in ("g.fa", "p.bed", "n.bed", "a.bw", "c.+.bw", "c.-.bw"):
		(tmp_path / name).write_text("")
	monkeypatch.chdir(tmp_path)

	with mock.patch("cherimoya_cli.commands.fit.run") as run:
		_main(monkeypatch, "fit", "name=demo", "sequences=g.fa",
			"loci=p.bed", "negatives=n.bed", "signals=[a.bw,[c.+.bw,c.-.bw]]",
			"n_filters=64")

	cfg = run.call_args.args[0]
	assert cfg.n_filters == 64
	assert list(cfg.signals[1]) == [str(tmp_path / "c.+.bw"),
		str(tmp_path / "c.-.bw")]


def test_pipeline_is_dispatched_to_hydra(tmp_path, monkeypatch):
	"""A step override reaches its node, and the shared inputs the steps
	interpolate are made absolute."""

	for name in ("g.fa", "p.bed", "s.bw"):
		(tmp_path / name).write_text("")
	monkeypatch.chdir(tmp_path)

	with mock.patch("cherimoya_cli.commands.pipeline.run") as run:
		_main(monkeypatch, "pipeline", "name=demo", "sequences=g.fa",
			"loci=[p.bed]", "negatives=null", "signals=[s.bw]",
			"fit.n_filters=64")

	cfg = run.call_args.args[0]
	assert cfg.fit.n_filters == 64
	assert cfg.negatives is None
	assert list(cfg.fit.loci) == [str(tmp_path / "p.bed")]
	assert cfg.attribute.sequences == str(tmp_path / "g.fa")


def test_cfg_job_prints_a_template_for_p(tmp_path, monkeypatch, capsys):
	"""`--cfg job` replaces `pipeline-json`: it prints the composed config
	without running anything, and the printed YAML runs as a `-p` file."""

	for name in ("g.fa", "a.bw"):
		(tmp_path / name).write_text("")
	monkeypatch.chdir(tmp_path)

	with mock.patch("cherimoya_cli.commands.pipeline.run") as run:
		_main(monkeypatch, "pipeline", "name=demo", "sequences=g.fa",
			"signals=[a.bw]", "loci=null", "negatives=null",
			"fit.n_filters=64", "--cfg", "job")
		run.assert_not_called()

		path = _yaml(tmp_path, capsys.readouterr().out)
		_main(monkeypatch, "pipeline", "-p", path)

	cfg = run.call_args.args[0]
	assert cfg.fit.n_filters == 64
	assert cfg.fit.name == "demo"
	assert cfg.loci is None


def test_multirun_jobs_get_their_own_directories(tmp_path, monkeypatch):
	"""Each job of a sweep runs in its own directory, so two jobs never
	write the same file, and relative inputs still point at the launch
	directory. The launch directory is restored afterwards."""

	import os

	from omegaconf import OmegaConf

	for name in ("g.fa", "p.bed", "n.bed", "s.bw"):
		(tmp_path / name).write_text("")
	monkeypatch.chdir(tmp_path)

	jobs = []

	def fake_fit(cfg):
		jobs.append((os.getcwd(), cfg.random_state, list(cfg.loci)))
		with open(cfg.name + ".torch", "w"):
			pass

	path = _yaml(tmp_path, "name: demo\nsequences: g.fa\nloci: [p.bed]\n"
		"negatives: [n.bed]\nsignals: [s.bw]\n")

	with mock.patch("cherimoya_cli.commands.fit.run", fake_fit):
		_main(monkeypatch, "fit", "-p", path, "-m", "random_state=0,1")

	assert os.getcwd() == str(tmp_path)

	(sweep,) = (tmp_path / "multirun").glob("*/*")
	assert sorted(jobs) == [
		(str(sweep / "0"), 0, [str(tmp_path / "p.bed")]),
		(str(sweep / "1"), 1, [str(tmp_path / "p.bed")]),
	]
	for num in ("0", "1"):
		assert (sweep / num / "demo.torch").exists()
		saved = OmegaConf.load(sweep / num / ".hydra" / "fit" / "config.yaml")
		assert saved.random_state == int(num)
	assert not (tmp_path / "demo.torch").exists()


def test_version(monkeypatch, capsys):
	from cherimoya_cli.__main__ import __version__

	with pytest.raises(SystemExit) as exit:
		_main(monkeypatch, "--version")

	assert exit.value.code == 0
	assert capsys.readouterr().out.strip() == "cherimoya " + __version__


def test_install_skill_still_uses_argparse(tmp_path, monkeypatch):
	with mock.patch("cherimoya_cli.commands.install_skill.run") as run:
		_main(monkeypatch, "install-skill", "-d", str(tmp_path), "--force")

	args = run.call_args.args[0]
	assert args.directory == str(tmp_path)
	assert args.force is True
