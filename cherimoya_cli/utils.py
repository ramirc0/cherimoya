# cherimoya_cli utilities
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import os


# The config keys that name each command's input files. A name is not an
# input everywhere: `attribute` writes the `ohe_filename`, `attr_filename`
# and `idx_filename` that `seqlets` reads.
INPUT_KEYS = {
	"negatives": ("peaks", "fasta", "bigwig"),
	"fit": ("sequences", "loci", "negatives", "signals", "controls",
		"exclusion_lists"),
	"evaluate": ("model", "sequences", "loci", "negatives", "signals",
		"controls", "exclusion_lists"),
	"attribute": ("model", "sequences", "loci", "exclusion_lists"),
	"seqlets": ("loci", "ohe_filename", "attr_filename", "idx_filename"),
	"marginalize": ("model", "sequences", "motifs", "loci", "exclusion_lists"),
	"pipeline": ("sequences", "loci", "negatives", "signals", "controls",
		"motifs", "exclusion_lists", "model"),
}


def draw_random_state():
	"""Draw a seed for a run whose `random_state` is null, and print it.

	A null `random_state` means "pick one and tell me" rather than "stay
	unseeded". The seed is printed whether or not `verbose` is set,
	because it is the one part of a run that cannot be recovered
	afterwards. Only rank 0 prints.

	The ranks Lightning launches after the first read rank 0's draw from
	`PL_GLOBAL_SEED`, which `seed_everything` sets before they start.
	Rank 0 itself always draws, so a value left in the environment by an
	earlier run in the same process is ignored. Under `srun` every rank
	starts at once, so there is no draw to inherit, and each derives the
	same seed from the job step instead.


	Returns
	-------
	random_state: int
		The drawn seed.
	"""

	import hashlib
	import os

	import numpy
	from lightning.pytorch.utilities import rank_zero_only

	say = rank_zero_only(print)

	seed = None
	if int(os.environ.get("LOCAL_RANK", 0)) > 0:
		seed = os.environ.get("PL_GLOBAL_SEED")
	elif int(os.environ.get("SLURM_NTASKS", 1)) > 1:
		step = "{}.{}".format(os.environ.get("SLURM_JOB_ID"),
			os.environ.get("SLURM_STEP_ID", 0))
		seed = int(hashlib.sha256(step.encode()).hexdigest(), 16) % (2**31 - 1)
		say("Derived random_state={0} from SLURM job step {1}; set "
			"random_state={0} to repeat this run.".format(seed, step))

	if seed is None:
		seed = int(numpy.random.randint(0, 2**31 - 1))
		say("Drew random_state={0}; set random_state={0} to repeat this "
			"run.".format(seed))

	return int(seed)


def resolve_inputs(cfg, keys):
	"""Make every local input path absolute and check that it exists.

	Relative paths resolve against the directory the command was launched
	from, so they still point at the right files after Hydra changes into
	a job directory. Remote paths (http://, https://, gs://, s3://) are
	left alone because checking them needs network access. The grouped
	`signals`/`controls` form (lists of lists) is walked recursively, and
	a null value stays null.


	Parameters
	----------
	cfg: omegaconf.DictConfig or dict
		A command's config. Its input keys are rewritten in place.

	keys: tuple of str
		The keys of `cfg` that name input files, such as
		`INPUT_KEYS["fit"]`.


	Raises
	------
	FileNotFoundError
		Listing every missing path, so all of them can be fixed at once.
	"""

	from hydra.utils import to_absolute_path

	missing = []

	def resolve(path, key):
		if path is None:
			return None
		if not isinstance(path, str):
			return [resolve(p, key) for p in path]
		if path.startswith(("http://", "https://", "gs://", "s3://")):
			return path

		path = to_absolute_path(path)
		if not os.path.exists(path):
			missing.append("{}: {}".format(key, path))
		return path

	for key in keys:
		cfg[key] = resolve(cfg[key], key)

	if missing:
		raise FileNotFoundError("The following inputs are missing:\n  - "
			+ "\n  - ".join(missing))
