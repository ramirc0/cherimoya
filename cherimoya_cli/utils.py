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
