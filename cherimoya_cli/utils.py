# cherimoya_cli utilities
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import copy
import os
import json
import dataclasses


# Config keys that name input files, across every command's schema.
INPUT_KEYS = ("sequences", "loci", "negatives", "signals", "controls",
	"motifs", "exclusion_lists", "model", "peaks", "fasta", "bigwig")


def _extract_set(parameters, defaults, name):
	subparameters = {
		key: parameters.get(key, None) for key in defaults if key in parameters
	}

	for parameter, value in parameters[name].items():
		if value is not None:
			subparameters[parameter] = value

	return subparameters


def _check_set(parameters, parameter, value):
	if parameters.get(parameter, None) == None:
		parameters[parameter] = value


def merge_parameters(parameters, default_parameters):
	"""Merge the provided parameters with the default parameters.


	Parameters
	----------
	parameters: str
		Name of the JSON folder with the provided parameters

	default_parameters: dict
		The default parameters for the operation.


	Returns
	-------
	params: dict
		The merged set of parameters.
	"""

	if isinstance(parameters, str):
		if not os.path.exists(parameters):
			raise FileNotFoundError("Parameter file not found: '{}'"
				.format(parameters))

		with open(parameters, "r") as infile:
			parameters = json.load(infile)

	# Keys whose default is None and which may simply be left out.
	unset_parameters = ("controls", "warning_threshold", "early_stopping",
		"exclusion_lists", "loss_weights", "model", "motifs", "progress_bar")
	for parameter, value in default_parameters.items():
		if parameter not in parameters:
			if value is None and parameter not in unset_parameters:
				raise ValueError("Must provide value for '{}'. Set it to "
					"null if this step is supposed to produce it."
					.format(parameter))

			# A copy, so that a caller writing into a nested dict does not
			# change the defaults for the next run in the same process.
			parameters[parameter] = copy.deepcopy(value)

	return parameters


def _json_config(command, parameters):
	"""Build a Hydra command's config from a JSON-style parameter dict, for
	the JSON-driven `fit` and `pipeline`. Keys the schema lacks are
	dropped."""

	from omegaconf import OmegaConf

	from .config import SCHEMAS

	schema = SCHEMAS[command]
	keys = {f.name for f in dataclasses.fields(schema)}
	return OmegaConf.merge(OmegaConf.structured(schema),
		{key: value for key, value in parameters.items() if key in keys})


def resolve_inputs(cfg):
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
		A command's config. The keys in `INPUT_KEYS` that it has are
		rewritten in place.


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

	for key in INPUT_KEYS:
		if key in cfg:
			cfg[key] = resolve(cfg[key], key)

	if missing:
		raise FileNotFoundError("The following inputs are missing:\n  - "
			+ "\n  - ".join(missing))
