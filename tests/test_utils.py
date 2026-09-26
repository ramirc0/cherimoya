"""Tests for the cherimoya_cli utility helpers."""

import pytest


# --------- resolve_inputs --------------------------------------------------

def test_resolve_inputs_makes_relative_paths_absolute(tmp_path, monkeypatch):
	"""Paths resolve against the launch directory, so they survive a
	later change of directory, including inside grouped signals."""

	from cherimoya_cli.utils import resolve_inputs

	for name in ("g.fa", "a.bw", "p.bw", "m.bw"):
		(tmp_path / name).write_text("")
	monkeypatch.chdir(tmp_path)

	cfg = {"sequences": "g.fa", "signals": ["a.bw", ["p.bw", "m.bw"]],
		"controls": None, "output": "out.bed"}
	resolve_inputs(cfg)

	assert cfg["sequences"] == str(tmp_path / "g.fa")
	assert cfg["signals"] == [str(tmp_path / "a.bw"),
		[str(tmp_path / "p.bw"), str(tmp_path / "m.bw")]]
	assert cfg["controls"] is None
	assert cfg["output"] == "out.bed"


def test_resolve_inputs_lists_every_missing_path(tmp_path, monkeypatch):
	from cherimoya_cli.utils import resolve_inputs

	monkeypatch.chdir(tmp_path)

	with pytest.raises(FileNotFoundError) as error:
		resolve_inputs({"sequences": "g.fa", "loci": ["x.bed", "y.bed"]})

	message = str(error.value)
	assert "sequences: {}".format(tmp_path / "g.fa") in message
	assert "loci: {}".format(tmp_path / "x.bed") in message
	assert "loci: {}".format(tmp_path / "y.bed") in message


def test_resolve_inputs_leaves_remote_paths_alone():
	from cherimoya_cli.utils import resolve_inputs

	remote = ["http://a/s.bw", "https://a/s.bw", "gs://a/s.bw", "s3://a/s.bw"]
	cfg = {"signals": list(remote)}
	resolve_inputs(cfg)

	assert cfg["signals"] == remote


def test_resolve_inputs_rewrites_a_config(tmp_path, monkeypatch):
	"""A typed config takes the rewritten paths, list-of-lists included."""

	from omegaconf import OmegaConf

	from cherimoya_cli.config import EvaluateConfig
	from cherimoya_cli.utils import resolve_inputs

	for name in ("m.torch", "g.fa", "x.bed", "p.bw", "m.bw"):
		(tmp_path / name).write_text("")
	monkeypatch.chdir(tmp_path)

	cfg = OmegaConf.structured(EvaluateConfig(model="m.torch",
		sequences="g.fa", loci="x.bed", signals=[["p.bw", "m.bw"]]))
	resolve_inputs(cfg)

	assert cfg.model == str(tmp_path / "m.torch")
	assert cfg.signals == [[str(tmp_path / "p.bw"), str(tmp_path / "m.bw")]]
