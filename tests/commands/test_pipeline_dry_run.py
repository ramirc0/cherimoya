"""Tests that `dry_run` emits the per-step YAMLs without running anything.

`dry_run` is how a pipeline config is checked before committing hours of
GPU time to it, so it has to complete on exactly the configurations a
real run would use -- including the one with a motif database, which is
the common case and the only one that reaches the annotation step.
"""


##


def test_dry_run_with_motifs_writes_no_annotation_outputs(tmp_path,
		run_pipeline):
	"""A dry run emits the step YAMLs and nothing else -- in particular
	not the tomtom-lite outputs, which would otherwise be empty files
	standing in for real results."""

	run_pipeline(motifs=str(tmp_path / "m.meme"))

	assert not (tmp_path / "demo.seqlets_annotated.bed").exists()
	assert not (tmp_path / "demo.motif_seqlet_count.tsv").exists()


def test_dry_run_with_motifs_writes_the_step_yamls(tmp_path, run_pipeline):
	"""The point of a dry run: every per-step YAML lands on disk so the
	config can be inspected, and each one is a complete config for its
	single-step command."""

	from omegaconf import OmegaConf

	from cherimoya_cli.config import SCHEMAS

	run_pipeline(motifs=str(tmp_path / "m.meme"), model=None, negatives=None)

	for command in ("negatives", "fit", "attribute", "seqlets",
			"marginalize"):
		step = OmegaConf.load(tmp_path / "demo.{}.yaml".format(command))
		OmegaConf.merge(OmegaConf.structured(SCHEMAS[command]), step)
		assert set(step) == set(SCHEMAS[command].__dataclass_fields__), command


def test_dry_run_assigns_the_produced_files(tmp_path, run_pipeline):
	"""A produced negatives file and model reach the steps that read
	them."""

	from omegaconf import OmegaConf

	run_pipeline(motifs=str(tmp_path / "m.meme"), model=None, negatives=None)

	fit = OmegaConf.load(tmp_path / "demo.fit.yaml")
	assert fit.negatives == ["demo.negatives.bed"]
	for command in ("attribute", "marginalize"):
		step = OmegaConf.load(tmp_path / "demo.{}.yaml".format(command))
		assert step.model == "demo.torch", command


def test_dry_run_attribute_yaml_uses_deep_lift_shap_settings(tmp_path,
		run_pipeline):
	"""The pipeline's shared `batch_size` (512) is sized for inference and
	must not reach the attribute step; the step pins its own, and the seed
	comes from the top level."""

	from omegaconf import OmegaConf

	run_pipeline(motifs=str(tmp_path / "m.meme"), random_state=7)

	step = OmegaConf.load(tmp_path / "demo.attribute.yaml")

	assert step.algorithm == "deep_lift_shap"
	assert step.batch_size == 64
	assert step.group == 0
	assert step.n_shuffles == 20
	assert step.random_state == 7
	# The top-level `compile: true` is not inherited.
	assert step.compile is False


def test_dry_run_marginalizes_over_the_negatives(tmp_path, pipeline_config,
		monkeypatch):
	"""Motifs are inserted into background loci: the negatives, unless the
	step names its own. The top-level `loci` are the peaks."""

	from omegaconf import OmegaConf

	from cherimoya_cli.commands import pipeline

	monkeypatch.chdir(tmp_path)

	pipeline.run(pipeline_config(motifs=str(tmp_path / "m.meme")))
	step = OmegaConf.load(tmp_path / "demo.marginalize.yaml")
	assert step.loci == [str(tmp_path / "n.bed")]

	cfg = pipeline_config(motifs=str(tmp_path / "m.meme"))
	cfg.marginalize.loci = str(tmp_path / "x.bed")
	pipeline.run(cfg)
	step = OmegaConf.load(tmp_path / "demo.marginalize.yaml")
	assert step.loci == str(tmp_path / "x.bed")


def test_pipeline_accepts_a_single_peak_file_as_a_string(tmp_path,
		run_pipeline):
	"""`loci` may be a bare path. The negatives step takes the first peak
	file, `${loci.0}`, which a bare string does not have."""

	from unittest import mock

	with mock.patch("subprocess.run"), \
			mock.patch("cherimoya_cli.commands.negatives.run") as negatives_run, \
			mock.patch("cherimoya_cli.commands.fit.run"), \
			mock.patch("cherimoya_cli.commands.attribute.run"), \
			mock.patch("cherimoya_cli.commands.seqlets.run"):
		run_pipeline(loci=str(tmp_path / "x.bed"), negatives=None,
			dry_run=False)

	assert negatives_run.call_args.args[0].peaks == str(tmp_path / "x.bed")
