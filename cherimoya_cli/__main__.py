# cherimoya_cli entry point
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import argparse
import importlib
import sys
from importlib.metadata import version, PackageNotFoundError

desc = """A command-line tool for the training and usage of Cherimoya models."""

_help = """Must be one of 'pipeline-json', 'pipeline', 'negatives',
    'fit', 'evaluate', 'attribute', 'seqlets', 'marginalize', or
    'install-skill'."""

try:
	__version__ = version("cherimoya")
except PackageNotFoundError:
	__version__ = "0.0.0+unknown"

# Commands configured by Hydra; the rest still parse their own arguments.
HYDRA_COMMANDS = {
	"negatives": "Sample GC-matched negatives.",
	"evaluate": "Evaluate a trained Cherimoya model.",
	"attribute": "Calculate attributions using a trained Cherimoya model, "
		"with DeepLIFT/SHAP (default) or saturation mutagenesis.",
	"seqlets": "Identify seqlets from attributions.",
	"marginalize": "Run marginalizations given motifs.",
}


def _setup_parsers() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=desc)
	parser.add_argument(
		"--version",
		action="version",
		version="cherimoya {}".format(__version__),
	)
	subparsers = parser.add_subparsers(help=_help, required=True, dest="cmd")

	for command, description in HYDRA_COMMANDS.items():
		subparsers.add_parser(command, help=description)

	# Pipeline JSON
	pipeline_json_parser = subparsers.add_parser(
		"pipeline-json",
		help="Make a pipeline JSON file given the provided information.",
	)
	pipeline_json_parser.add_argument(
		"-s", "--sequences", type=str, required=True,
		help="The FASTA file of sequences."
	)
	pipeline_json_parser.add_argument(
		"-i",
		"--inputs",
		type=str,
		action="append",
		required=True,
		help="A BAM or bigwig file. Repeatable.",
	)
	pipeline_json_parser.add_argument(
		"-c",
		"--controls",
		type=str,
		action="append",
		help="A BAM or bigwig file. Repeatable.",
	)
	pipeline_json_parser.add_argument(
		"-p",
		"--peaks",
		type=str,
		action="append",
		help="A BED-formatted file of peaks to use. Repeatable.",
	)
	pipeline_json_parser.add_argument(
		"-neg",
		"--negatives",
		type=str,
		action="append",
		help="A BED-formatted file of negative loci to use. Repeatable.",
	)
	pipeline_json_parser.add_argument(
		"-n", "--name", type=str, required=True,
		help="Name to use as a suffix in intermediary files."
	)
	pipeline_json_parser.add_argument(
		"-u",
		"--unstranded",
		action="store_true",
		default=False,
		help="Whether the input is unstranded.",
	)
	pipeline_json_parser.add_argument(
		"-f",
		"--fragments",
		action="store_true",
		default=False,
		help="Whether the input are fragments or reads.",
	)
	pipeline_json_parser.add_argument(
		"-ps",
		"--pos_shift",
		type=int,
		default=0,
		help="How many bp to shift the + strand reads.",
	)
	pipeline_json_parser.add_argument(
		"-ns",
		"--neg_shift",
		type=int,
		default=0,
		help="how many bp to shift the - strand reads.",
	)
	pipeline_json_parser.add_argument(
		"-m",
		"--motifs",
		type=str,
		default=None,
		help="A motif database for marginalization and TF-MoDISco.",
	)
	pipeline_json_parser.add_argument(
		"-o", "--output", type=str, required=True,
		help="The filename for the pipeline JSON."
	)
	pipeline_json_parser.add_argument(
		"-pe",
		"--paired_end",
		action="store_true",
		default=False,
		help="Whether the input is paired-end.",
	)
	pipeline_json_parser.add_argument(
		"-sf",
		"--scale_factor",
		type=float,
		default=1,
		help="Whether to scale the read counts. 1 is no scaling.",
	)

	# Fit
	fit_parser = subparsers.add_parser("fit", help="Fit a Cherimoya model.")
	fit_parser.add_argument(
		"-p",
		"--parameters",
		type=str,
		required=True,
		help="A JSON file containing the parameters for fitting the model.",
	)

	# Pipeline
	pipeline_parser = subparsers.add_parser(
		"pipeline", help="Run each step on the given files."
	)
	pipeline_parser.add_argument(
		"-p",
		"--parameters",
		type=str,
		required=True,
		help="A JSON file containing the parameters used for each step.",
	)

	# Install skill
	install_skill_parser = subparsers.add_parser(
		"install-skill",
		help="Install the bundled Cherimoya agent skill for Claude Code.",
	)
	install_skill_parser.add_argument(
		"-d",
		"--directory",
		type=str,
		default=None,
		help="Skills directory to install into. Default is ~/.claude/skills.",
	)
	install_skill_parser.add_argument(
		"--symlink",
		action="store_true",
		default=False,
		help="Symlink the packaged skill instead of copying it. Reflects "
		"in-place edits, but breaks if the install location moves.",
	)
	install_skill_parser.add_argument(
		"-f",
		"--force",
		action="store_true",
		default=False,
		help="Overwrite an existing installation at the destination.",
	)

	return parser


def _task(command):
	def task(cfg):
		from .config import missing_keys
		from .utils import resolve_inputs

		missing = missing_keys(cfg)
		if missing:
			raise ValueError("Must provide a value for: {}".format(
				", ".join(sorted(missing))))

		resolve_inputs(cfg)
		importlib.import_module(".commands." + command, __package__).run(cfg)

	return task


def _dispatch(command, argv):
	"""Run a Hydra command. `-p FILE` layers a YAML file over the command's
	schema, so the file is type-checked and `key=value` overrides still
	apply on top of it. Hydra reserves `-p` for `--package`, so it is
	consumed here before Hydra parses the rest."""

	import hydra
	from hydra.core.config_store import ConfigStore
	from omegaconf import OmegaConf

	from . import config  # noqa: F401 -- registers the schemas

	if "-p" in argv:
		i = argv.index("-p")
		ConfigStore.instance().store(group="run", name="user",
			node=OmegaConf.load(argv[i + 1]), package="_global_")
		argv = argv[:i] + argv[i + 2:] + ["+run=user"]

	sys.argv = ["cherimoya " + command] + argv
	hydra.main(version_base="1.3", config_path="pkg://cherimoya_cli.conf",
		config_name=command)(_task(command))()


def main():
	"""Entry point of the `cherimoya` console script."""

	if len(sys.argv) > 1 and sys.argv[1] in HYDRA_COMMANDS:
		return _dispatch(sys.argv[1], sys.argv[2:])

	parser = _setup_parsers()
	args = parser.parse_args()

	COMMANDS = {
		"pipeline-json": "pipeline_json",
		"fit": "fit",
		"pipeline": "pipeline",
		"install-skill": "install_skill",
	}

	mod = importlib.import_module(f".commands.{COMMANDS[args.cmd]}",
		package=__package__)
	mod.run(args)


if __name__ == "__main__":
	main()
