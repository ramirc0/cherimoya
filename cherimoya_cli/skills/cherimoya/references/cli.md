# CLI subcommand map

`cherimoya` has one subcommand per pipeline stage. Most people only need
`pipeline` (see `references/cli-training-pipeline.md`); the rest rerun a single
stage. This maps a goal to a subcommand. For exhaustive key tables, defer to
<https://cherimoya.readthedocs.io/en/latest/cli.html>.

## Goal to subcommand

| Goal | Subcommand |
|---|---|
| Run the whole thing end to end | `pipeline` |
| Just train (and auto-evaluate) a model | `fit` |
| Score an existing model on held-out data | `evaluate` |
| Compute per-base attributions (DeepLIFT/SHAP, or ISM via `algorithm`) | `attribute` |
| Call seqlets from attributions | `seqlets` |
| Measure a model's response to inserted motifs | `marginalize` |
| Sample GC-matched negative regions | `negatives` |
| Install this skill for Claude Code | `install-skill` |

## One calling convention

```
cherimoya <command> [-p FILE] [key=value ...] [-m] [Hydra flags]
```

- Every subcommand except `install-skill` takes the same form. `-p` is always
  a YAML config file; `key=value` overrides go on top of it. There are no
  per-command flags.
- `cherimoya <command> --help` prints the command's full config with defaults.
  `???` marks a required key.
- `--cfg job` prints the composed config and runs nothing. Redirect it to a
  file to make a `-p` template.
- Unknown keys are errors. Every missing required key is listed in one error.
- Quote list values: `'signals=[a.bw,b.bw]'`.
- `-m` sweeps comma-separated values into separate jobs (see
  `references/cli-training-pipeline.md`).
- JSON configs from earlier versions don't load. Rewrite them as YAML.

`fit`, `evaluate`, `attribute`, `seqlets` and `marginalize` take `skip=true`
to no-op the step. In `pipeline`, a top-level `skip=true` no-ops the whole run
and `annotation.skip=true` skips seqlet annotation. The MoDISco steps and
`negatives` have no `skip`. Inputs are still checked first, so a missing file
fails even when skipped. `pipeline` also takes `dry_run=true`, which writes the
per-step YAML files and runs nothing.

## `negatives` (standalone)

```bash
cherimoya negatives peaks=peaks.narrowPeak fasta=genome.fa output=negatives.bed
```

`peaks`, `fasta` and `output` are required. Optional: `bigwig` (with `beta`,
drops negatives with too much signal), `bin_width` (0.02), `max_n_perc` (0.1),
`beta` (0.5), `in_window` (2114), `out_window` (1000), `verbose` (false).

## Key defaults (from `config.py`)

Quote these; don't guess others. Read `cherimoya_cli/config.py` or the docs.

- Windows: `in_window` 2114, `out_window` 1000.
- Model: `n_filters` 128, `n_layers` 9, `expansion` 2.
- Training: `batch_size` 64, `max_epochs` 20, `min_total_steps` 20000,
  `early_stopping` `null` (off; set an int for epochs of no validation
  count-Pearson improvement), `n_warmup_epochs` 2, `negative_ratio` 0.25,
  `reverse_complement` true, `num_workers` 1 (per device), `loss_weights`
  `null`, `random_state` 0, `devices` 1 (`-1` for every visible device; more
  than one trains with DDP and splits the global `batch_size` evenly across
  them).
- Training output: `verbose` false for a standalone `fit` (the pipeline's
  top-level `verbose`, true by default, reaches its fit step; true prints the
  run setup and the per-epoch table of `{name}.log`), `progress_bar` `null`
  (with `verbose`, draw Lightning's progress bar only when stdout is a terminal
  or a Jupyter kernel; `true`/`false` force it).
- Inference: `evaluate` and `marginalize` use `batch_size` 512; `attribute`
  uses 64.
- Device/dtype: `cuda` / `float32`.
- Split (hg38): `validation_chroms` = chr8, chr20 (choose the checkpoint).
  `test_chroms` = chr1, chr3, chr6 (scored once after training). Everything
  else trains. **Change all three for non-hg38 genomes.**
- Peak calling: `preprocessing.callpeaks_gsize` `hs`, `preprocessing.callpeaks_q`
  0.05.
- Seqlets: `threshold` 0.01, lengths 4 to 25 bp, `additional_flanks` 3.

## Keys that do nothing

Some declared keys are never read. Don't suggest them as fixes:

- `seqlets.verbose`.

## `min_total_steps`: an epoch is not a fixed amount of training

An epoch is one pass over the peaks, so `max_epochs` alone buys a step count
proportional to how many peaks an experiment has. 20 epochs is 280 optimizer
steps for an experiment with 14 batches of peaks and 54,000 for one with
2,700. `min_total_steps` (default 20000) raises `max_epochs` at run time until
the run reaches that many steps, and the warmup and cosine decay are laid out
over the raised value. Set it to `null` for a deliberately short run, e.g. a
smoke test: `max_epochs=1 min_total_steps=null`.

## `loss_weights`: fixed weights instead of the learned ones

By default the two loss terms are balanced by Kendall uncertainty weighting:
learned parameters `lw0` and `lw1`, one per signal group, carried by a third
optimizer and frozen mid-run by a gradient threshold. Setting `loss_weights`
replaces them with constants:

```bash
cherimoya fit -p run.yaml 'loss_weights=[1.333,0.274]'
```

or in a YAML file:

```yaml
loss_weights: [1.333, 0.274]
```

**Use these values.** They reproduce the operating point the learned weights
reach (`lw0` 7.0 to 8.0, `lw1` 1.2 to 1.5, measured across both TF and
accessibility runs), so the fixed scheme trains to the same place without the
extra optimizer. `1.333` multiplies the profile MNLL **after it is divided by
each signal group's own batch-mean read depth**; `0.274` multiplies the count
MSE as-is.

Two things follow from that asymmetry and are worth knowing before changing
the numbers:

- The **profile** term is a sum of per-read log-likelihoods, so it scales with
  read depth. That is what the division handles, and why it is per group
  rather than pooled. One divisor for the whole batch would rescale every
  group equally and leave their weights relative to each other untouched.
- The **count** term is computed on `log1p` counts, where depth is an additive
  shift the model absorbs into its bias. It needs no depth normalization.
  Normalizing it by the *variance* of log counts instead is worse, not better:
  that variance is largely the signal being predicted.

When `loss_weights` is set, `lw0` and `lw1` stop receiving gradient, the
`log(lw)²` prior is dropped, and the freeze rule does not run. They remain on
the model, so checkpoints are unaffected either way. `null` (the default)
keeps the learned weights. Under `verbose`, `fit` prints `Fixed Loss
Weights: profile=..., count=...` in place of the `SGD Optimizer (lw)` line,
so the log says which of the two schemes the run used.
