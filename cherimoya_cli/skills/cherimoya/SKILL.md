---
name: cherimoya
description: >-
  Train, evaluate, and use Cherimoya sequence-to-function genomic models
  with the `cherimoya` command-line tools or Python API. Use when a user 
  wants to run the end-to-end pipeline, use a trained model to do
  downstream tasks, or troubleshoot a run. Designed for users who may be 
  new to Cherimoya or to sequence modeling: ask clarifying questions 
  whenever an input is ambiguous or missing, pick reasonable defaults, 
  and explain in plain language what was chosen and why.
---

# Cherimoya

Cherimoya is a compact deep learning model that predicts genomic profile data —
transcription-factor binding, chromatin accessibility, transcription initiation —
directly from DNA sequence. It ships an end-to-end command-line pipeline that takes
raw sequencing data through peak calling, training, attribution, and motif
discovery in a couple of commands, and a small Python API for programmatic use.

This skill helps you drive that pipeline for someone who may not know the
terminology. **Most of the value here is behavioral, not encyclopedic:** be
robust, ask before guessing, and always explain what a default did.

## How to use this skill

1. Read the three operating rules below — they apply to *every* task.
2. Figure out what the user actually wants (train? evaluate? interpret results?
   troubleshoot?) and which inputs they have.
3. Open only the reference file that matches the task (table at the bottom).
   Do not preload all of them.

## Three operating rules (always in effect)

### 1. Ask, don't guess — for anything biologically meaningful

A wrong genome build, strandedness, or read-shift produces a model that trains
without error and is quietly wrong. So before running anything, confirm the
things that can't be recovered from a filename:

- **What is the assay?** (ATAC-seq, DNase-seq, ChIP-seq, CUT&RUN, PRO-seq, …)
  This drives stranded-vs-unstranded, fragment-vs-read, and read shifts. See
  `references/assay-defaults.md`.
- **What genome build are the reads aligned to?** (hg38, hg19, mm10, …) The
  FASTA, the peaks, and the default chromosome split must all match it.
- **Which files does the user actually have**, and what is each one? Never
  invent a path. See `references/input-files.md`.
- **What is the goal?** "Is my data any good?" → train + read
  `{name}.test.performance.tsv`. "What motifs did it learn?" → attributions + seqlets +
  TF-MoDISco. Map lay language onto subcommands before acting.

When in doubt, ask a short question. One clarifying question is always cheaper
than a wasted training run.

### 2. Handle missing files by asking, not by inventing

The required inputs for training are a **genome FASTA** and at least one
**signal file**. Peaks, negatives, controls, and a motif database are optional
(the pipeline generates or skips them). Before building a pipeline command:

- List which required inputs are present and which are missing.
- If a required input is missing, **ask the user for it**. Don't fabricate a
  path or silently proceed.
- If an *optional* input is missing, say what the pipeline will do instead
  (e.g. "no peaks given, so MACS3 will call them", "no negatives given, so
  GC-matched negatives will be sampled") so the user can veto it.
- Every command checks its local input paths before running and fails with a
  list of every missing one. Your job is to catch this *earlier* by asking.
  Remote paths (`http://`, `https://`, `s3://`, `gs://`) are not checked and
  are streamed directly.
- `loci` and `negatives` are required keys even when the user has no file.
  Set them to `null` to have the pipeline make them (`loci=null` calls peaks,
  `negatives=null` samples negatives). Leaving the key out is an error, and so
  is pointing it at a file that doesn't exist yet.

### 3. Use reasonable defaults and explain them

Every default lives in the schemas in `cherimoya_cli/config.py`, and
`cherimoya <command> --help` prints them. Quote the real value, never a guessed
one. Whenever a default or an automatic step kicks in, tell the user in one
plain sentence what happened and why. Examples:

- "You didn't pass peaks, so MACS3 will call them at q < 0.05 (its default)."
- "Training will use chr8 and chr20 as held-out validation, the hg38 default.
  If your data isn't hg38, we need to change this."
- "The model trains for at least 20 epochs, more if that is under 20,000
  steps (early stopping is off by default), and
  the checkpoint kept is the epoch with the best validation count Pearson."

The point is that a novice should never be surprised by something the pipeline
did on their behalf.

## The common request: "train a model on my data"

The end-to-end flow is one command, best kept in a config file (details in
`references/cli-training-pipeline.md`):

```bash
# 1. Write the full config, with every default, to a file. Runs nothing.
cherimoya pipeline name=my_run sequences=genome.fa loci=null negatives=null \
    'signals=[signal.bam]' motifs=motifs.meme --cfg job > my_run.yaml

# 2. Run every step from that file. key=value overrides still apply on top.
cherimoya pipeline -p my_run.yaml
```

Before step 1, walk rule 1 and rule 2: confirm the assay, the genome build,
and which inputs exist. Ask how many GPUs the run may use. With more than
one, training can split each batch across them (the "Several GPUs" edit in
`references/cli-training-pipeline.md`). Assay settings are `preprocessing.*`
keys (see `references/assay-defaults.md`). Quote list values so the shell
leaves the brackets alone. Step 2 runs every stage through to
marginalization. Some stages are conditional: tomtom-lite seqlet annotation
and marginalization run only when `motifs` is set. See
`references/cli-training-pipeline.md` for the per-step table and what gates
each step.

Configs from older Cherimoya versions were JSON. `-p` still reads one if
every key is in the new schema, but a pipeline JSON needs converting by
hand first. `references/cli-training-pipeline.md` lists the steps.

## Reference map — open the one that fits the task

| The user wants to… | Read |
|---|---|
| Understand a term or how the model works (profile vs counts, seqlets, EMA, …) | `references/concepts.md` |
| Train / run the full pipeline from raw data | `references/cli-training-pipeline.md` |
| Train on several GPUs, watch training progress | `references/cli-training-pipeline.md` |
| Know what file types they have / need | `references/input-files.md` |
| Set assay-specific options (shifts, strandedness) | `references/assay-defaults.md` |
| Understand what the pipeline produced | `references/interpreting-outputs.md` |
| Fix an error or a bad-looking result | `references/troubleshooting.md` |
| Run one subcommand or find a config key | `references/cli.md` |
| Save / load / run inference with the model in Python, or train from a Python script | `references/using-in-python.md` |
| Attribute (DeepLIFT/SHAP, ISM), design, score variants | `references/using-tangermeme.md` |

Full project documentation lives at <https://cherimoya.readthedocs.io> (glossary,
architecture, per-assay recipes, API reference). When a detail isn't in these
reference files, defer to the docs rather than guessing.
