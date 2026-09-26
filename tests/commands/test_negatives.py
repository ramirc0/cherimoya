"""Wiring tests for `cherimoya negatives`.

The GC matching itself is tangermeme's; what is Cherimoya's is the
config surface and the forwarding, which is what these cover.
"""

from unittest import mock

import pandas
from omegaconf import OmegaConf

from cherimoya_cli.config import NegativesConfig, missing_keys


def _config(**overrides):
	return OmegaConf.structured(NegativesConfig(**{"peaks": "peaks.bed",
		"fasta": "g.fa", "output": "out.bed", **overrides}))


def _captured(cfg, tmp_path):
	from cherimoya_cli.commands import negatives

	captured = {}
	frame = pandas.DataFrame([["chr1", 100, 200]])

	def fake_match(**kwargs):
		captured.update(kwargs)
		return frame

	cfg.output = str(tmp_path / "out.bed")
	with mock.patch("tangermeme.match.extract_matching_loci",
			side_effect=fake_match):
		negatives.run(cfg)

	captured["_written"] = pandas.read_csv(cfg.output, sep="\t",
		header=None)
	return captured


##


def test_negatives_forwards_every_flag(tmp_path):
	"""Each config key has to reach `extract_matching_loci` under the name
	that function expects, which is not the key's own name for most of
	them."""

	captured = _captured(_config(bin_width=0.05, max_n_perc=0.2, beta=0.7,
		in_window=1000, out_window=500), tmp_path)

	assert captured["gc_bin_width"] == 0.05
	assert captured["max_n_perc"] == 0.2
	assert captured["signal_beta"] == 0.7
	assert captured["in_window"] == 1000
	assert captured["out_window"] == 500


def test_negatives_writes_a_headerless_bed(tmp_path):
	"""The output feeds straight into `fit` as a locus file, so it must
	be headerless and tab separated."""

	captured = _captured(_config(), tmp_path)

	assert list(captured["_written"].iloc[0]) == ["chr1", 100, 200]
	assert len(captured["_written"].columns) == 3


def test_negatives_bigwig_is_optional(tmp_path):
	"""`bigwig` sets a minimum-counts threshold; without it the
	threshold is off rather than the call failing."""

	captured = _captured(_config(bigwig=None), tmp_path)

	assert captured["bigwig"] is None


def test_negatives_requires_peaks_fasta_and_output(make_config):
	"""The three keys have no default, so a config without them is
	rejected before the command runs rather than failing later."""

	assert missing_keys(make_config("negatives")) == {"peaks", "fasta",
		"output"}


def test_negatives_defaults_match_the_documented_ones(make_config):
	"""The defaults the pipeline's negative sampling used to hard-code,
	now shared with the command."""

	cfg = make_config("negatives")

	assert cfg.bin_width == 0.02
	assert cfg.max_n_perc == 0.1
	assert cfg.beta == 0.5
	assert cfg.in_window == 2114
	assert cfg.out_window == 1000
	assert cfg.bigwig is None
	assert cfg.verbose is False
