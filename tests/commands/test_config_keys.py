"""Tests that every declared config key is one something reads.

A key that nothing reads is worse than no key: it appears in the saved
config and in the CLI reference, so setting it looks like it should do
something. Two have already shipped that way -- marginalize's
`output_folder`, which the command spells `output_filename`, and fit's
`count_loss_weight`, which nothing has ever read -- so the invariant is
pinned for every step rather than for the two that happened to be found.
"""

import dataclasses
import pathlib
import re
import typing

import pytest

from cherimoya_cli import config as C


# The pipeline's step nodes and the schema that types each one. A key
# the schema lacks is rejected when the config is composed.
STEP_SCHEMAS = {
	"preprocessing": C.PreprocessConfig,
	"negative_sampling": C.NegativesConfig,
	"fit": C.FitConfig,
	"attribute": C.AttributeConfig,
	"seqlets": C.SeqletsConfig,
	"annotation": C.AnnotationConfig,
	"modisco_motifs": C.ModiscoMotifsConfig,
	"modisco_report": C.ModiscoReportConfig,
	"marginalize": C.MarginalizeConfig,
}

# The module that reads each schema. The pipeline-only steps are read by
# the pipeline itself.
READERS = {
	C.NegativesConfig: "negatives",
	C.FitConfig: "fit",
	C.EvaluateConfig: "evaluate",
	C.AttributeConfig: "attribute",
	C.SeqletsConfig: "seqlets",
	C.MarginalizeConfig: "marginalize",
	C.PreprocessConfig: "pipeline",
	C.AnnotationConfig: "pipeline",
	C.ModiscoMotifsConfig: "pipeline",
	C.ModiscoReportConfig: "pipeline",
}

##


def test_every_pipeline_step_is_typed_by_its_schema():
	hints = typing.get_type_hints(C.PipelineConfig)
	assert {step: hints[step] for step in STEP_SCHEMAS} == STEP_SCHEMAS


@pytest.mark.parametrize("schema", list(READERS), ids=lambda s: s.__name__)
def test_every_declared_key_is_read(schema):
	"""A text check: each key must appear as `"key"` or `.key` in the
	module that reads the schema."""

	path = pathlib.Path(C.__file__).parent / "commands" / (READERS[schema]
		+ ".py")
	source = path.read_text()

	unread = {field.name for field in dataclasses.fields(schema)
		if not re.search(r"""["']{0}["']|\.{0}\b""".format(field.name), source)}
	if schema is C.FitConfig:
		# fit hands the evaluation after training every key the two share.
		unread -= {field.name for field in dataclasses.fields(C.EvaluateConfig)}

	assert not unread


def test_marginalize_output_key_is_declared():
	"""The other half of the rename: dropping the key entirely would
	leave the checks above passing while the output path became
	unsettable."""

	keys = {field.name for field in dataclasses.fields(C.MarginalizeConfig)}
	assert "output_filename" in keys
	assert "output_folder" not in keys
