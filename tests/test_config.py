"""Tests for the Hydra config schemas in cherimoya_cli.config."""

import copy

import pytest
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

import cherimoya_cli.defaults as D
from cherimoya_cli.config import missing_keys
from cherimoya_cli.utils import _check_set, _extract_set, merge_parameters


REQUIRED = {
	"negatives": ["peaks=p.bed", "fasta=g.fa", "output=n.bed"],
	"fit": ["name=demo", "sequences=g.fa", "loci=[p.bed]", "negatives=[n.bed]",
		"signals=[s.bw]"],
	"evaluate": ["model=m.torch", "sequences=g.fa", "loci=[p.bed]",
		"signals=[s.bw]"],
	"attribute": ["model=m.torch", "sequences=g.fa", "loci=[p.bed]"],
	"seqlets": ["loci=[p.bed]", "ohe_filename=o.npz", "attr_filename=a.npz",
		"idx_filename=i.npy"],
	"marginalize": ["model=m.torch", "sequences=g.fa", "motifs=m.meme",
		"loci=[n.bed]"],
	"pipeline": ["name=demo", "sequences=g.fa", "loci=[p.bed]",
		"negatives=[n.bed]", "signals=[s.bw]"],
}


def _resolved(node):
	return OmegaConf.to_container(node, resolve=True)


@pytest.mark.parametrize("command", sorted(REQUIRED))
def test_every_command_composes_with_its_required_keys(make_config, command):
	cfg = make_config(command, *REQUIRED[command])
	assert missing_keys(cfg) == set()


def test_missing_required_keys_are_reported_together(make_config):
	cfg = make_config("fit")
	assert missing_keys(cfg) == {"name", "sequences", "loci",
		"negatives", "signals"}


def test_a_key_the_pipeline_produces_may_be_null(make_config):
	cfg = make_config("pipeline", "name=demo", "sequences=g.fa", "loci=null",
		"negatives=null", "signals=[s.bw]")
	assert missing_keys(cfg) == set()
	assert cfg.loci is None and cfg.negatives is None


def test_steps_do_not_hide_the_missing_shared_keys(make_config):
	cfg = make_config("pipeline", "sequences=g.fa", "loci=null",
		"negatives=null")
	assert missing_keys(cfg) == {"name", "signals"}


def test_a_key_that_cannot_be_produced_rejects_null(make_config):
	with pytest.raises(ConfigCompositionException):
		make_config("fit", "sequences=null")


@pytest.mark.parametrize("override", ["n_filter=64", "fit.n_filter=64"])
def test_an_unknown_key_is_rejected(make_config, override):
	command = "pipeline" if override.startswith("fit.") else "fit"
	with pytest.raises(ConfigCompositionException):
		make_config(command, override)


def test_a_value_of_the_wrong_type_is_rejected(make_config):
	with pytest.raises(ConfigCompositionException):
		make_config("fit", "n_filters=abc")


def test_grouped_signals_keep_their_nesting(make_config):
	cfg = make_config("fit", "signals=[atac.bw,[c.+.bw,c.-.bw]]")
	assert _resolved(cfg)["signals"] == ["atac.bw", ["c.+.bw", "c.-.bw"]]


def test_a_single_run_writes_into_the_working_directory():
	import cherimoya_cli.config  # noqa: F401
	from hydra import compose, initialize_config_module

	with initialize_config_module("cherimoya_cli.conf", version_base="1.3"):
		cfg = compose("fit", return_hydra_config=True)

	assert cfg.hydra.run.dir == "."
	assert cfg.hydra.output_subdir == ".hydra/fit"


# --------- parity with the JSON defaults ----------------------------------
#
# Transitional: these compare against cherimoya_cli.defaults and the JSON
# merge helpers, and go away with them.

JSON_DEFAULTS = {
	"fit": D.default_fit_parameters,
	"evaluate": D.default_evaluate_parameters,
	"attribute": D.default_attribute_parameters,
	"seqlets": D.default_seqlet_parameters,
	"marginalize": D.default_marginalize_parameters,
}


@pytest.mark.parametrize("command", sorted(JSON_DEFAULTS))
def test_schema_defaults_match_the_json_defaults(make_config, command):
	cfg = make_config(command, *REQUIRED[command])
	schema = OmegaConf.to_container(cfg)
	defaults = JSON_DEFAULTS[command]

	# evaluate reads `negatives`, which the JSON defaults do not declare.
	assert set(schema) - set(defaults) <= {"negatives"}
	assert set(defaults) <= set(schema)

	for key, value in defaults.items():
		if value is not None:
			assert schema[key] == value, key


def _json_pipeline():
	parameters = copy.deepcopy(D.default_pipeline_parameters)
	parameters.update(name="demo", sequences="g.fa", loci=["p.bed"],
		negatives=["n.bed"], signals=["s.bw"], motifs="m.meme", model="m.torch",
		random_state=7)
	return parameters


def _json_step(parameters, step):
	"""Build a step's parameters the way the JSON pipeline does."""

	pname = parameters["name"]
	if step == "fit":
		sub = _extract_set(parameters, D.default_fit_parameters, "fit_parameters")
		return merge_parameters(sub, D.default_fit_parameters)

	if step == "attribute":
		sub = _extract_set(parameters, D.default_attribute_parameters,
			"attribute_parameters")
		_check_set(sub, "ohe_filename", pname + ".attributions.ohe.npz")
		_check_set(sub, "attr_filename", pname + ".attributions.attr.npz")
		_check_set(sub, "idx_filename", pname + ".attributions.idxs.npy")
		return merge_parameters(sub, D.default_attribute_parameters)

	if step == "seqlets":
		attribute = _json_step(parameters, "attribute")
		sub = _extract_set(parameters, D.default_seqlet_parameters,
			"seqlet_parameters")
		_check_set(sub, "ohe_filename", pname + ".attributions.ohe.npz")
		_check_set(sub, "attr_filename", pname + ".attributions.attr.npz")
		_check_set(sub, "idx_filename", pname + ".attributions.idxs.npy")
		_check_set(sub, "output_filename", pname + ".seqlets.bed")
		_check_set(sub, "chroms", attribute["chroms"])
		return merge_parameters(sub, D.default_seqlet_parameters)

	sub = _extract_set(parameters, D.default_marginalize_parameters,
		"marginalize_parameters")
	sub["loci"] = parameters["marginalize_parameters"]["loci"] or parameters[
		"negatives"]
	_check_set(sub, "output_filename", pname + "_marginalize/")
	_check_set(sub, "motifs", parameters["motifs"])
	return merge_parameters(sub, D.default_marginalize_parameters)


@pytest.mark.parametrize("step", ["fit", "attribute", "seqlets", "marginalize"])
def test_pipeline_steps_match_the_json_pipeline(make_config, step):
	# `model` is set as it is by the time attribute and marginalize run.
	cfg = make_config("pipeline", *REQUIRED["pipeline"], "motifs=m.meme",
		"model=m.torch", "random_state=7")

	assert _resolved(cfg[step]) == _json_step(_json_pipeline(), step)


def test_pipeline_annotation_and_modisco_match_the_json_pipeline(make_config):
	cfg = make_config("pipeline", *REQUIRED["pipeline"], "motifs=m.meme")

	assert _resolved(cfg.annotation) == dict(
		D.default_annotation_parameters, sequences="g.fa", motifs="m.meme",
		seqlet_filename="demo.seqlets.bed",
		output_filename="demo.seqlets_annotated.bed")
	assert _resolved(cfg.modisco_motifs) == {"n_seqlets": 100000,
		"output_filename": "demo_modisco_results.h5", "verbose": True}
	assert _resolved(cfg.modisco_report) == {"motifs": "m.meme",
		"output_folder": "demo_modisco/", "verbose": True}


# --------- pipeline interpolation -----------------------------------------

def test_a_step_override_changes_only_that_step(make_config):
	cfg = make_config("pipeline", *REQUIRED["pipeline"], "fit.in_window=3000")
	assert cfg.fit.in_window == 3000
	assert cfg.attribute.in_window == 2114


def test_a_shared_key_set_after_composing_reaches_the_steps(make_config):
	cfg = make_config("pipeline", "name=demo", "sequences=g.fa", "loci=null",
		"negatives=null", "signals=[s.bw]")
	cfg.loci = ["demo_peaks.narrowPeak"]

	assert list(cfg.fit.loci) == ["demo_peaks.narrowPeak"]
	assert cfg.negative_sampling.peaks == "demo_peaks.narrowPeak"
