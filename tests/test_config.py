"""Tests for the Hydra config schemas in cherimoya_cli.config."""

import pytest
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from cherimoya_cli.config import missing_keys


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


# --------- what each pipeline step inherits -------------------------------

def _pipeline(make_config):
	# `model` is set as it is by the time attribute and marginalize run.
	return make_config("pipeline", *REQUIRED["pipeline"], "motifs=m.meme",
		"model=m.torch", "random_state=7")


def test_pipeline_fit_names_its_outputs_after_the_run(make_config):
	fit = _resolved(_pipeline(make_config).fit)

	assert fit["name"] == "demo"
	assert fit["random_state"] == 7
	# The shared `batch_size` (512) is sized for inference.
	assert fit["batch_size"] == 64


def test_pipeline_seqlets_read_what_attribute_writes(make_config):
	cfg = _resolved(_pipeline(make_config))
	attribute, seqlets = cfg["attribute"], cfg["seqlets"]

	assert attribute["ohe_filename"] == "demo.attributions.ohe.npz"
	assert attribute["attr_filename"] == "demo.attributions.attr.npz"
	assert attribute["idx_filename"] == "demo.attributions.idxs.npy"
	for key in ("ohe_filename", "attr_filename", "idx_filename", "chroms"):
		assert seqlets[key] == attribute[key], key
	assert seqlets["output_filename"] == "demo.seqlets.bed"


def test_pipeline_marginalize_inherits_the_shared_keys(make_config):
	marginalize = _resolved(_pipeline(make_config).marginalize)

	assert marginalize["model"] == "m.torch"
	assert marginalize["motifs"] == "m.meme"
	assert marginalize["output_filename"] == "demo_marginalize/"
	assert marginalize["batch_size"] == 512
	assert marginalize["random_state"] == 7


def test_pipeline_marginalize_runs_on_the_negatives(make_config):
	"""Motifs are inserted into background loci, so marginalize takes the
	negatives unless `marginalize.loci` names its own."""

	cfg = _pipeline(make_config)
	assert OmegaConf.to_container(cfg.marginalize)["loci"] == "${negatives}"
	assert _resolved(cfg.marginalize)["loci"] == ["n.bed"]


def test_pipeline_annotation_and_modisco(make_config):
	cfg = make_config("pipeline", *REQUIRED["pipeline"], "motifs=m.meme")

	assert _resolved(cfg.annotation) == {"sequences": "g.fa",
		"seqlet_filename": "demo.seqlets.bed", "motifs": "m.meme",
		"n_score_bins": 100, "n_median_bins": 1000, "n_target_bins": 100,
		"n_cache": 250, "reverse_complement": True, "n_jobs": -1,
		"output_filename": "demo.seqlets_annotated.bed", "skip": False}
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
