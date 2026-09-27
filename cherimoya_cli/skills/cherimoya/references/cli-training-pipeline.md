# Training a model with the CLI pipeline

`cherimoya pipeline` runs every step, from raw signal files to motifs, in one
command. This is the path for "train a Cherimoya model on my data."

First complete the checks in `SKILL.md` (assay, genome build, which inputs
exist). This file assumes that's done.

## How a command is configured

Every command except `install-skill` is configured with Hydra. A config comes
from three layers, each overriding the one before:

1. The schema defaults in `cherimoya_cli/config.py`.
2. An optional YAML file passed with `-p FILE`.
3. `key=value` overrides on the command line.

Unknown keys are errors, in the file and in overrides, so a typo such as
`n_filter=64` fails before any work starts.

Override syntax:

- Nested keys are dotted: `fit.n_filters=64`, `preprocessing.unstranded=true`.
- Quote lists, or the shell expands the brackets: `'signals=[a.bw,b.bw]'`,
  `'signals=[[ctcf.+.bw,ctcf.-.bw]]'`.
- `null` sets a key to null: `negatives=null`.
- `${key}` refers to another key. Quote it: `'signals=[${name}.bw]'`.
- Hydra reads the overrides as one unbroken run. Flags such as `-m` and
  `--cfg` go before or after all of them, never between. `-p FILE` may go
  anywhere.

JSON parameter files from earlier versions don't load, and there is no
converter. Write the same keys as YAML and drop the `_parameters` suffix from
the step sections (`fit_parameters` becomes `fit`).

## Step 1: the five required keys

The pipeline needs `name`, `sequences`, `loci`, `negatives` and `signals`. A
missing one stops the run with one error that lists all of them:

```
Must provide a value for: loci, negatives, sequences, signals. Set a key to null if an earlier pipeline step produces it.
```

`null` means "an earlier step produces this". `loci=null` calls peaks with
MACS3, and `negatives=null` samples GC-matched negatives. Leaving the key out
is not the same as `null`. It's an error.

TF ChIP-seq with controls and no peaks yet:

```bash
cherimoya pipeline name=my_run sequences=genome.fa \
    loci=null negatives=null \
    'signals=[chip_rep1.bam,chip_rep2.bam]' \
    'controls=[input_rep1.bam,input_rep2.bam]' \
    motifs=motifs.meme
```

With a peak file, pass `'loci=[peaks.narrowPeak]'` instead of `loci=null`.
`controls`, `motifs`, `model` and `exclusion_lists` are optional and default to
`null`.

Assay-specific keys (`preprocessing.unstranded`, `preprocessing.fragments`,
`preprocessing.paired_end`, `preprocessing.pos_shift`/`neg_shift`,
`preprocessing.scale_factor`) are in `references/assay-defaults.md`. Get them
right, because they change the biology, not just the bookkeeping.

## Step 2 (recommended): write the config to a file

`--cfg job` prints the full composed config as YAML and runs nothing. Use it to
make a file the user can read and keep:

```bash
cherimoya pipeline name=my_run sequences=genome.fa loci=null negatives=null \
    'signals=[chip.bam]' 'controls=[input.bam]' motifs=motifs.meme \
    --cfg job > my_run.yaml
```

Keys still printed as `???` are required and have no value yet. Fill them in.
Step values such as `name: ${name}` or `loci: ${loci}` link a step to the
top-level key, so changing `name` or `in_window` at the top reaches every step.
`--cfg` doesn't work together with `-m`.

Common novice edits. Always explain the change you make.

- **Not hg38?** `fit.training_chroms`, `fit.validation_chroms` and
  `fit.test_chroms` default to hg38 names. Update all three for the user's
  genome, or validation is silently empty. `fit` refuses to start if two of
  them share a chromosome, and `fit.test_chroms=null` skips the test
  evaluation. For mouse, also set `preprocessing.callpeaks_gsize` to `mm`.
- **Out of GPU memory?** Lower `fit.batch_size` (64 to 32 or 16), lower
  `fit.n_filters` (128 to 64), or set `dtype=bfloat16`.
- **Several GPUs?** Set `fit.devices` (default 1, or `-1` for every visible
  GPU) to train with DDP. `batch_size` is the *global* batch, split evenly
  across devices, so it must be divisible by `devices`. Each step sees the
  same examples as on one GPU, and steps, epochs and the schedule mean the
  same thing. Tell the user:
  - GPUs are the first `devices` visible ones. Pick them with
    `CUDA_VISIBLE_DEVICES=2,3 cherimoya pipeline -p ...` on a shared machine.
  - Each GPU holds `batch_size / devices` examples, so the global batch can
    grow with the GPU count. Host memory does not shrink: every GPU's process
    loads the whole training set and runs its own `num_workers` workers.
  - Results match a one-GPU run statistically, not bitwise.
  - It must run from a terminal or batch script, not a Jupyter notebook.
  - Under SLURM, run `srun cherimoya fit -p my_run.fit.yaml devices=4` (not
    `pipeline`) with `--ntasks-per-node` equal to `devices` on one node. With
    `random_state=null`, every rank derives the same seed from the SLURM job
    step and rank 0 prints it.
  The pipeline runs the fit step as a separate `python -m cherimoya_cli fit`
  process. Details: <https://cherimoya.readthedocs.io/en/latest/cli.html#training-on-several-devices>.
- **Want to watch training?** The pipeline's top-level `verbose` (true by
  default) reaches the fit step, which then prints the per-epoch table (the
  rows of `{name}.log`) above Lightning's progress bar, with the latest
  validation Pearsons to the bar's right. Several runs sharing one terminal
  overwrite each other's bars: set `fit.progress_bar=false` for them.
  Redirected to a file, the bar is off by default.
- **Small dataset?** See the dataset-size guidance in
  `references/troubleshooting.md`.
- **Want replicates?** `random_state` defaults to `0`, so rerunning the same
  config rebuilds the *same* model rather than an independent replicate. Give
  each replicate its own seed, or sweep it (below).
- **Skip a step:** set that step's `skip=true`, e.g. `attribute.skip=true`.
  `fit`, `attribute`, `seqlets`, `annotation` and `marginalize` take it. A
  top-level `skip=true` skips the whole pipeline.
- **Dry run:** `dry_run=true` writes every per-step YAML file and runs nothing.
  It's a cheap way to show the user what will run before spending GPU time.

Every key and default is tabulated at
<https://cherimoya.readthedocs.io/en/latest/cli.html>.

## Step 3: run it

```bash
cherimoya pipeline -p my_run.yaml
cherimoya pipeline -p my_run.yaml fit.n_filters=64   # with a one-off change
```

Before any work, every local input path is made absolute (relative paths
resolve against the directory the command started in) and checked. All
missing files are listed together:

```
FileNotFoundError: The following inputs are missing:
  - sequences: /data/run/genome.fa
```

Remote paths (`http://`, `https://`, `s3://`, `gs://`) are not checked; they
are streamed. If a missing path is something the pipeline will create later,
set that key to `null`.

## What the pipeline does, in order

Each step reads its own section of the config (`preprocessing`,
`negative_sampling`, `fit`, `attribute`, `seqlets`, `annotation`,
`modisco_motifs`, `modisco_report`, `marginalize`). Before the negatives, fit,
attribute, seqlets and marginalize steps, the pipeline saves that step's
section, links filled in, as `{name}.<command>.yaml`. The preprocessing,
annotation and MoDISco steps write no config file.

| Step | Runs when | Produces |
|---|---|---|
| MACS3 peak calling | `loci` is `null` | `{name}_peaks.narrowPeak` |
| `bam2bw` signal to bigWig | signals aren't already bigWig (`.sam/.bam/.bed[.gz]/.tsv[.gz]`) | `{name}.+.bw`/`{name}.-.bw` (stranded) or `{name}.bw` (unstranded); controls give `{name}.control.*.bw` |
| negative sampling | `negatives` is `null` | `{name}.negatives.bed` |
| train, then evaluate | `model` is `null` | `{name}.torch`, `{name}.final.torch`, `{name}.log`, `{name}.detailed.log`, `{name}.{validation,test}.performance.tsv` |
| attribute (DeepLIFT/SHAP) | always | `{name}.attributions.{ohe,attr}.npz`, `{name}.attributions.idxs.npy` |
| seqlets | always | `{name}.seqlets.bed` |
| tomtom-lite annotation | `motifs` is set | `{name}.seqlets_annotated.bed`, `{name}.motif_seqlet_count.tsv` |
| TF-MoDISco | always | `{name}_modisco_results.h5`, `{name}_modisco/` |
| marginalization | `motifs` is set | `{name}_marginalize/` |

Hydra also saves the composed config and the overrides in `.hydra/pipeline/`.

Worth telling a user up front:

- **Set `model` to an existing checkpoint and training is skipped.** That
  checkpoint is used for every later step.
- **TF-MoDISco runs even without a motif database.** It discovers motifs
  *de novo*; the database only *names* them. Without `motifs`, annotation and
  marginalization are skipped, so the run ends after TF-MoDISco.
- **Marginalization inserts motifs into the negatives**, not into the
  peaks.
- `{name}.torch` is the best-validation checkpoint the later steps load. See
  `references/interpreting-outputs.md` for `.torch` vs `.final.torch`.
- **Seeds.** With `random_state=null`, the pipeline draws one seed before any
  step, prints `Drew random_state=N; set random_state=N to repeat this run.`
  and saves it in every step's YAML file, so fit, attribute and marginalize
  share it.

## Rerunning a single step

The per-step YAML files are complete configs for the single-step commands:

```bash
cherimoya fit         -p my_run.fit.yaml
cherimoya evaluate    -p my_run.test.evaluate.yaml
cherimoya attribute   -p my_run.attribute.yaml
cherimoya seqlets     -p my_run.seqlets.yaml
cherimoya marginalize -p my_run.marginalize.yaml
cherimoya negatives   -p my_run.negatives.yaml
```

Overrides work here too, so a changed rerun needs no editing:

```bash
cherimoya seqlets -p my_run.seqlets.yaml threshold=0.001 \
    output_filename=my_run.strict.seqlets.bed
```

Things that make rerunning work:

- After conversion, the pipeline points `signals`/`controls` at the produced
  bigWigs, so `my_run.fit.yaml` trains from the bigWigs even though the input
  was a BAM.
- `fit` runs `evaluate` twice after training, on the validation and the test
  chromosomes, and writes `my_run.validation.evaluate.yaml` and
  `my_run.test.evaluate.yaml` first. There's no separate evaluate step in the
  pipeline, but you can rerun either standalone.
- The evaluations after training take the pipeline's `compile` and
  `compile_mode`, as training does. On a `torch.compile` error, rerun one with
  `cherimoya evaluate -p my_run.test.evaluate.yaml compile=false`.

## Training many models

`-m` runs one job per value, or per combination of values:

```bash
cherimoya fit -p my_run.fit.yaml -m random_state=0,1,2
cherimoya fit -p my_run.fit.yaml -m n_filters=64,128 random_state=0,1
```

Each job runs in its own `multirun/<date>/<time>/<n>/` directory with its own
outputs and `.hydra/fit/`, so jobs never overwrite each other. Relative input
paths still resolve against the directory the sweep started in.

By default the jobs run one after another. To submit each job to SLURM,
install the `slurm` extra (`pip install "cherimoya[slurm]"`) and pick the
submitit launcher:

```bash
cherimoya fit -p my_run.fit.yaml -m random_state=0,1,2 \
    hydra/launcher=submitit_slurm hydra.launcher.partition=gpu \
    hydra.launcher.gpus_per_node=1 hydra.launcher.timeout_min=240
```

Ask the user for the partition name and time limit; they are cluster-specific.
Launcher logs go to `.submitit/` inside the sweep directory, and `squeue` lists
the jobs as `cherimoya-fit` (or `cherimoya-<command>`). Adding `--cfg
hydra` (without `-m`) prints every launcher setting.
