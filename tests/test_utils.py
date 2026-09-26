"""Tests for the cherimoya_cli utility helpers."""

import json

import pytest

from cherimoya_cli.utils import _check_set, _extract_set, merge_parameters


# --------- merge_parameters -----------------------------------------------

def test_merge_fills_in_missing_defaults(tmp_path):
	defaults = {'a': 1, 'b': 2, 'c': 3}
	user = {'a': 10}
	path = tmp_path / "p.json"
	path.write_text(json.dumps(user))
	out = merge_parameters(str(path), defaults)
	assert out == {'a': 10, 'b': 2, 'c': 3}


def test_merge_accepts_dict_directly():
	defaults = {'a': 1, 'b': 2}
	out = merge_parameters({'a': 5}, defaults)
	assert out['a'] == 5 and out['b'] == 2


def test_merge_raises_on_missing_required(tmp_path):
	"""Defaults of None are treated as required (with a small whitelist of
	exceptions like 'controls' and 'exclusion_lists')."""

	defaults = {'sequences': None, 'a': 1}
	path = tmp_path / "p.json"
	path.write_text(json.dumps({}))
	with pytest.raises(ValueError, match="sequences"):
		merge_parameters(str(path), defaults)


def test_merge_allows_none_for_unset_parameters(tmp_path):
	"""`controls`, `exclusion_lists`, `early_stopping`, etc. may be None."""

	defaults = {'controls': None, 'exclusion_lists': None}
	path = tmp_path / "p.json"
	path.write_text(json.dumps({}))
	out = merge_parameters(str(path), defaults)
	assert out['controls'] is None
	assert out['exclusion_lists'] is None


def test_merge_raises_on_missing_file():
	with pytest.raises(FileNotFoundError):
		merge_parameters("/no/such/file.json", {'a': 1})


# --------- _check_set ------------------------------------------------------

def test_check_set_only_writes_when_missing():
	d = {'a': 5}
	_check_set(d, 'a', 99)
	_check_set(d, 'b', 99)
	assert d == {'a': 5, 'b': 99}


def test_check_set_treats_none_as_missing():
	d = {'a': None}
	_check_set(d, 'a', 7)
	assert d == {'a': 7}


# --------- _extract_set ----------------------------------------------------

def test_extract_set_combines_top_level_with_subdict_overrides():
	defaults = {'in_window': 100, 'out_window': 50}
	parameters = {
		'in_window': 200,
		'out_window': 80,
		'fit_parameters': {
			'in_window': None,        # falls through to top-level (200)
			'out_window': 40,         # explicit override (40)
		},
	}
	out = _extract_set(parameters, defaults, 'fit_parameters')
	assert out['in_window'] == 200
	assert out['out_window'] == 40


def test_merge_parameters_does_not_share_the_default_dicts():
	"""A caller that writes into a filled-in nested dict, as the pipeline
	does, must not change the defaults the next run starts from."""

	from cherimoya_cli.utils import merge_parameters

	defaults = {"nested": {"output_filename": None}, "flag": 1}
	merged = merge_parameters({}, defaults)
	merged["nested"]["output_filename"] = "first_run.h5"

	assert defaults["nested"]["output_filename"] is None


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
