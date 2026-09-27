# cherimoya_cli config schemas
# Author: Jacob Schreiber <jmschreiber91@gmail.com>
#
# One dataclass per step. Hydra composes a command's config from its schema,
# so every key has a type and a default, and an unknown key is an error.
#
# A field set to MISSING must be given. `Optional[...] = MISSING` means it
# must be given but may be null; in the pipeline, null means "an earlier
# step produces this". Fields typed `Any` take either a path or a list of
# paths; `signals`/`controls` also take the grouped list-of-lists form.
#
# `PipelineConfig` gives each step a node whose defaults interpolate the
# shared top-level keys, so `name=x` or `in_window=3000` reaches every step
# and `fit.batch_size=32` changes only one.
#
# A note on `signals` / `controls`:
#
# These accept either a flat list of file paths or a structured list
# whose entries are each ``str`` (a one-channel unstranded group) or
# ``list[str]`` (a multi-channel group such as a stranded ``(+, -)``
# pair). The grouping decides how reverse-complement augmentation
# permutes channels and how many predictions the count head emits.
# Examples::
#
#     signals: [atac.bw]
#         # one unstranded group, one count prediction
#
#     signals: [[ctcf.+.bw, ctcf.-.bw]]
#         # one stranded group, one count prediction shared across +/-
#
#     signals: [atac.bw, [ctcf.+.bw, ctcf.-.bw]]
#         # one unstranded group + one stranded group; two count predictions
#
# A flat list of N files is interpreted as N independent unstranded
# groups. BPNet-style callers that pass ``[plus.bw, minus.bw]`` as a
# stranded pair must use the nested form ``[[plus.bw, minus.bw]]`` to keep
# the +/- swap on RC.
#
# A note on `random_state`:
#
# Training is seeded by default so that a run can be repeated: the seed
# fixes the model's initialization and the peak/negative sampler's draw
# order. Set it to any other integer to get an independent run -- rerunning
# the same config unchanged reproduces the same model rather than giving an
# independent replicate. Setting it to null does not turn seeding off; it
# means "draw a seed and print it". The pipeline draws one seed for every
# step and records it in each step's `<name>.<command>.yaml`.
#
# Seeding does not make CUDA training bitwise reproducible. The fused
# conv+norm kernel reduces with relaxed atomics, so the summation order
# varies between launches, and two same-seed GPU runs diverge as training
# compounds that difference. What the seed fixes is the initialization and
# the sequence of examples, not the arithmetic. CPU runs with the same seed
# are bitwise identical.

from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf


training_chroms = ["chr2", "chr4", "chr5", "chr7", "chr9", "chr10", "chr11",
	"chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19",
	"chr21", "chr22", "chrX", "chrY"]

validation_chroms = ["chr8", "chr20"]

# Held out of both lists above, and scored once after training to give a
# performance estimate that did not take part in choosing the checkpoint.
test_chroms = ["chr1", "chr3", "chr6"]


def _chroms(chroms):
	return field(default_factory=lambda: list(chroms))


@dataclass
class NegativesConfig:
	"""Parameters for sampling GC-matched negatives (`cherimoya negatives`)."""

	peaks: str = MISSING
	fasta: str = MISSING
	output: str = MISSING
	bigwig: Optional[str] = None
	bin_width: float = 0.02
	max_n_perc: float = 0.1
	beta: float = 0.5
	in_window: int = 2114
	out_window: int = 1000
	verbose: bool = False


@dataclass
class FitConfig:
	"""Parameters for training a model (`cherimoya fit`)."""

	name: Optional[str] = MISSING
	sequences: str = MISSING
	loci: Any = MISSING
	negatives: Any = MISSING
	signals: Any = MISSING
	controls: Any = None
	exclusion_lists: Any = None
	training_chroms: List[str] = _chroms(training_chroms)
	validation_chroms: List[str] = _chroms(validation_chroms)
	test_chroms: Optional[List[str]] = _chroms(test_chroms)
	n_filters: int = 128
	n_layers: int = 9
	expansion: int = 2
	residual_scale: float = 0.15
	batch_size: int = 64
	in_window: int = 2114
	out_window: int = 1000
	max_jitter: int = 500
	reverse_complement: bool = True
	reverse_complement_average: bool = False
	summits: bool = False
	max_epochs: int = 20
	min_total_steps: Optional[int] = 20000
	loss_weights: Optional[List[float]] = None
	muon_lr: float = 0.025
	muon_wd: float = 0.03
	adam_lr: float = 0.001
	adam_wd: float = 0.0
	lw_lr: float = 0.001
	lw_wd: float = 0.0
	lw_momentum: float = 0.9
	n_warmup_epochs: int = 2
	negative_ratio: float = 0.25
	num_workers: int = 1
	early_stopping: Optional[int] = None
	random_state: Optional[int] = 0
	compile: bool = True
	compile_mode: str = "max-autotune"
	dtype: str = "float32"
	device: str = "cuda"
	devices: int = 1
	progress_bar: Optional[bool] = None
	verbose: bool = False
	skip: bool = False


@dataclass
class EvaluateConfig:
	"""Parameters for evaluating a trained model (`cherimoya evaluate`)."""

	model: str = MISSING
	sequences: str = MISSING
	loci: Any = MISSING
	negatives: Any = None
	signals: Any = MISSING
	controls: Any = None
	exclusion_lists: Any = None
	chroms: List[str] = _chroms(validation_chroms)
	batch_size: int = 512
	in_window: int = 2114
	out_window: int = 1000
	reverse_complement_average: bool = False
	summits: bool = False
	compile: bool = True
	compile_mode: str = "max-autotune"
	dtype: str = "float32"
	device: str = "cuda"
	verbose: bool = False
	performance_filename: str = "performance.tsv"
	skip: bool = False


@dataclass
class AttributeConfig:
	"""Parameters for calculating attributions (`cherimoya attribute`)."""

	model: str = MISSING
	sequences: str = MISSING
	loci: Any = MISSING
	exclusion_lists: Any = None
	chroms: List[str] = _chroms(training_chroms + validation_chroms)
	algorithm: str = "deep_lift_shap"
	output: str = "counts"
	group: Optional[int] = 0
	attr_window: int = 400
	n_shuffles: int = 20
	warning_threshold: Optional[float] = 1e-3
	random_state: Optional[int] = 0
	batch_size: int = 64
	in_window: int = 2114
	compile: bool = False
	compile_mode: str = "max-autotune"
	dtype: str = "float32"
	device: str = "cuda"
	verbose: bool = False
	ohe_filename: str = "attributions.ohe.npz"
	attr_filename: str = "attributions.attr.npz"
	idx_filename: str = "attributions.idx.npy"
	skip: bool = False


@dataclass
class SeqletsConfig:
	"""Parameters for calling seqlets from attributions (`cherimoya seqlets`)."""

	loci: Any = MISSING
	ohe_filename: str = MISSING
	attr_filename: str = MISSING
	idx_filename: str = MISSING
	chroms: List[str] = _chroms(training_chroms + validation_chroms)
	threshold: float = 0.01
	min_seqlet_len: int = 4
	max_seqlet_len: int = 25
	additional_flanks: int = 3
	verbose: bool = False
	output_filename: str = "seqlets.bed"
	skip: bool = False


@dataclass
class MarginalizeConfig:
	"""Parameters for motif marginalization (`cherimoya marginalize`)."""

	model: str = MISSING
	sequences: str = MISSING
	motifs: str = MISSING
	loci: Any = MISSING
	exclusion_lists: Any = None
	chroms: List[str] = _chroms(training_chroms)
	n_loci: Optional[int] = 100
	shuffle: bool = False
	attributions: bool = False
	minimal: bool = True
	random_state: Optional[int] = 0
	batch_size: int = 512
	in_window: int = 2114
	compile: bool = True
	compile_mode: str = "max-autotune"
	device: str = "cuda"
	verbose: bool = False
	output_filename: str = "marginalize/"
	skip: bool = False


@dataclass
class PreprocessConfig:
	"""Pipeline step 0: peak calling with MACS3 and conversion with bam2bw."""

	unstranded: bool = False
	fragments: bool = False
	paired_end: bool = False
	pos_shift: int = 0
	neg_shift: int = 0
	scale_factor: float = 1.0
	read_depth: bool = False
	callpeaks_format: Optional[str] = None
	callpeaks_gsize: str = "hs"
	callpeaks_q: float = 0.05
	verbose: bool = True


@dataclass
class AnnotationConfig:
	"""Pipeline step 3.2: annotating seqlets against a motif database."""

	sequences: str = MISSING
	seqlet_filename: str = MISSING
	motifs: Optional[str] = None
	n_score_bins: int = 100
	n_median_bins: int = 1000
	n_target_bins: int = 100
	n_cache: int = 250
	reverse_complement: bool = True
	n_jobs: int = -1
	output_filename: str = "seqlets_annotated.bed"
	skip: bool = False


@dataclass
class ModiscoMotifsConfig:
	"""Pipeline step 4.1: `modisco motifs`."""

	output_filename: str = MISSING
	n_seqlets: int = 100000
	verbose: bool = False


@dataclass
class ModiscoReportConfig:
	"""Pipeline step 4.2: `modisco report`."""

	output_folder: str = MISSING
	motifs: Optional[str] = None
	verbose: bool = False


def _step(cls, **interpolations):
	return field(default_factory=lambda: cls(**interpolations))


@dataclass
class PipelineConfig:
	"""Parameters for every step of `cherimoya pipeline`.

	The top-level keys are shared. Setting `loci`, `negatives` or `model`
	to null makes the pipeline produce it: peaks are called with MACS3,
	negatives are sampled, and a model is trained.
	"""

	name: str = MISSING
	sequences: str = MISSING
	loci: Any = MISSING
	negatives: Any = MISSING
	signals: Any = MISSING
	controls: Any = None
	exclusion_lists: Any = None
	motifs: Optional[str] = None
	model: Optional[str] = None
	in_window: int = 2114
	out_window: int = 1000
	batch_size: int = 512
	random_state: Optional[int] = 0
	compile: bool = True
	compile_mode: str = "max-autotune"
	dtype: str = "float32"
	device: str = "cuda"
	verbose: bool = True
	skip: bool = False
	dry_run: bool = False

	preprocessing: PreprocessConfig = field(default_factory=PreprocessConfig)

	negative_sampling: NegativesConfig = _step(NegativesConfig,
		peaks="${loci.0}", fasta="${sequences}",
		output="${name}.negatives.bed", in_window="${in_window}",
		out_window="${out_window}", verbose="${preprocessing.verbose}")

	fit: FitConfig = _step(FitConfig,
		name="${name}", sequences="${sequences}", loci="${loci}",
		negatives="${negatives}", signals="${signals}",
		controls="${controls}", exclusion_lists="${exclusion_lists}",
		in_window="${in_window}", out_window="${out_window}",
		random_state="${random_state}", compile="${compile}",
		compile_mode="${compile_mode}", dtype="${dtype}",
		device="${device}", verbose="${verbose}")

	attribute: AttributeConfig = _step(AttributeConfig,
		model="${model}", sequences="${sequences}", loci="${loci}",
		exclusion_lists="${exclusion_lists}", in_window="${in_window}",
		random_state="${random_state}", compile_mode="${compile_mode}",
		dtype="${dtype}", device="${device}", verbose="${verbose}",
		ohe_filename="${name}.attributions.ohe.npz",
		attr_filename="${name}.attributions.attr.npz",
		idx_filename="${name}.attributions.idxs.npy")

	seqlets: SeqletsConfig = _step(SeqletsConfig,
		loci="${loci}", chroms="${attribute.chroms}",
		ohe_filename="${attribute.ohe_filename}",
		attr_filename="${attribute.attr_filename}",
		idx_filename="${attribute.idx_filename}",
		verbose="${verbose}", output_filename="${name}.seqlets.bed")

	annotation: AnnotationConfig = _step(AnnotationConfig,
		sequences="${sequences}", motifs="${motifs}",
		seqlet_filename="${seqlets.output_filename}",
		output_filename="${name}.seqlets_annotated.bed")

	modisco_motifs: ModiscoMotifsConfig = _step(ModiscoMotifsConfig,
		output_filename="${name}_modisco_results.h5", verbose="${verbose}")

	modisco_report: ModiscoReportConfig = _step(ModiscoReportConfig,
		output_folder="${name}_modisco/", motifs="${motifs}",
		verbose="${verbose}")

	marginalize: MarginalizeConfig = _step(MarginalizeConfig,
		model="${model}", sequences="${sequences}", motifs="${motifs}",
		loci="${negatives}", exclusion_lists="${exclusion_lists}",
		in_window="${in_window}", batch_size="${batch_size}",
		random_state="${random_state}",
		compile="${compile}", compile_mode="${compile_mode}",
		device="${device}", verbose="${verbose}",
		output_filename="${name}_marginalize/")


def missing_keys(cfg):
	"""Return the dotted names of the required keys that were not given.

	Unlike `OmegaConf.missing_keys`, this does not resolve interpolations,
	so a pipeline step pointing at a key an earlier step produces, such as
	`${loci.0}` while `loci` is null, is not reported.


	Parameters
	----------
	cfg: omegaconf.DictConfig
		A composed config.


	Returns
	-------
	missing: set of str
		The keys still set to MISSING (`???`).
	"""

	def walk(node, prefix):
		for key, value in node.items():
			if value == MISSING:
				yield prefix + key
			elif isinstance(value, dict):
				yield from walk(value, prefix + key + ".")

	return set(walk(OmegaConf.to_container(cfg), ""))


SCHEMAS = {
	"negatives": NegativesConfig,
	"fit": FitConfig,
	"evaluate": EvaluateConfig,
	"attribute": AttributeConfig,
	"seqlets": SeqletsConfig,
	"marginalize": MarginalizeConfig,
	"pipeline": PipelineConfig,
}

for _name, _schema in SCHEMAS.items():
	ConfigStore.instance().store(name="{}_schema".format(_name), node=_schema)
