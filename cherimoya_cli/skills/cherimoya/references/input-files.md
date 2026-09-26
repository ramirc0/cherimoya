# Identifying the user's input files

Before building a pipeline you need to know *what the user actually has*. This
is a recognition guide: given a file (often just an extension), what is it, which
pipeline key does it fill, and what to confirm. For assay-driven settings
(shifts, strandedness), see `references/assay-defaults.md`.

## The pipeline slots

A training run has, at most, these inputs:

| Slot | Pipeline key | Required? |
|---|---|---|
| Reference genome | `sequences` | **Yes** |
| Signal | `signals` (list) | **Yes** |
| Control | `controls` (list) | No |
| Peaks | `loci` (list) | Key required; `null` has MACS3 call them |
| Negatives | `negatives` (list) | Key required; `null` samples them |
| Motif database | `motifs` | No; enables annotation and marginalization |
| Exclusion list | `exclusion_lists` (list) | No; BED of regions to drop from train/validation |

If a required slot is empty, ask the user for it. For each empty optional slot,
tell the user what the pipeline will do instead. Set `exclusion_lists` to mask
regions, e.g. the ENCODE blacklist: `'exclusion_lists=[blacklist.bed]'`.

## Recognizing file types

### Reference genome: `sequences`
- **`.fa`, `.fasta`, `.fa.gz`**: the genome the reads were aligned to.
- Must match the **genome build** of the signal and peaks (hg38 ≠ hg19 ≠ mm10).
  The single most important thing to confirm; a mismatch trains a
  silently wrong model.
- Chromosome names in the FASTA must match `fit.training_chroms` /
  `fit.validation_chroms` (e.g. `chr8` vs `8`).

### Signal: `signals`
The measured coverage the model learns to predict. Two forms:
- **Aligned reads or fragments: `.bam`, `.sam`, `.bed`, `.bed.gz`, `.tsv`,
  `.tsv.gz`**. `bam2bw` converts them to bigWig. Fragment files (10x-style)
  need `preprocessing.fragments=true`.
- **Coverage tracks: `.bw`, `.bigwig`**. Already processed; used directly and
  conversion is skipped.

Confirm with the user:
- **Reads or fragments?** A `.bam` almost always holds **aligned reads**; a
  `.tsv`/`.tsv.gz` (often `.bed`/`.bed.gz`) usually holds **fragments**. This
  decides `preprocessing.fragments` and `preprocessing.paired_end`. See the
  reads-vs-fragments table in `references/assay-defaults.md`.
- **Stranded or unstranded?** ChIP/ATAC/DNase are usually unstranded (one
  track); many TF/initiation assays are stranded (`+`/`-` pair). This decides
  `preprocessing.unstranded` and the grouping (below). See
  `references/assay-defaults.md`.
- **Replicates?** How they combine depends on input type. **BAM/SAM/fragment**
  replicates listed together in `signals` are merged into one bigWig by
  `bam2bw` (one pooled track). **Pre-made bigWig** replicates are *not* pooled.
  A flat list is read as N independent unstranded groups (see the grouping
  footgun below), so pool them upstream first. Peak calling pools all
  replicates as MACS3 treatments in both cases.

> **Check keys against the filetype.** A mismatch is silent:
> - `fragments=true` with a `.bam` of aligned reads, or a fragment `.tsv`
>   *without* it, misparses the input.
> - `paired_end` only does something for a paired-end read BAM (sets MACS3's
>   `BAMPE`); it is **ignored for fragment files**, so `fragments` together with
>   `paired_end` is a contradiction. The intent was `fragments` alone.
>
> If the keys and filetype disagree, stop and ask which is correct.

### Control: `controls`
- Same file types as signal. For ChIP-seq this is the **input / IgG control**
  (unenriched DNA). A model trained *with* controls must be evaluated and used
  *with* the same controls, or the count head sees garbage.

### Peaks: `loci`
- **`.narrowPeak`, `.bed`, `.broadPeak`**: genomic regions of interest
  (positives). With `loci=null`, MACS3 calls them from the signal at q < 0.05.
- `fit.summits=true` centers loci on the narrowPeak summit column. It needs a
  true narrowPeak (10-column) file. Don't set `summits` with a `.broadPeak` or
  plain `.bed`, which have no summit column.

### Negatives: `negatives`
- **`.bed`**: background regions the model also trains on. With
  `negatives=null`, the pipeline samples GC-matched negatives from the first
  `loci` file. `fit.negative_ratio` (default 0.25) sets negatives per peak per
  epoch. Every negative on the validation chromosomes is also scored at
  validation and evaluation, for the peaks+negatives count metrics and the
  AUROC/AUPRC.

### Motif database: `motifs`
- **`.meme`**: MEME-format known motifs (e.g. JASPAR, HOCOMOCO). Optional;
  passing it enables tomtom-lite seqlet annotation and marginalization and lets
  the TF-MoDISco report *name* discovered motifs. Without it, TF-MoDISco still
  discovers motifs *de novo* but leaves them unnamed.

## The stranded-signal footgun (important)

`signals` (and `controls`) accept a flat or nested list, and the shape changes
the meaning:

- **Flat list**: each entry is its own **independent unstranded** track.
  `[a.bw, b.bw]` is two separate unstranded outputs.
- **Nested list**: the inner list is one **multi-channel group**, e.g. a
  stranded `(+, -)` pair. `[[ctcf.+.bw, ctcf.-.bw]]` is one stranded group.

Getting this wrong is silent. A stranded pair written flat as
`[plus.bw, minus.bw]` becomes two unstranded tracks and disables the `+/-`
swap during reverse-complement augmentation. Wrap stranded pairs, quoted on the
command line as `'signals=[[plus.bw,minus.bw]]'`, or in YAML:

```yaml
signals:
  - [plus.bw, minus.bw]
```

When the pipeline converts BAMs itself it writes the grouped form for you; you
assemble it by hand only for pre-made bigWigs. Full semantics: the header
comment in `cherimoya_cli/config.py`.

## Remote files

Any path can be a remote URL (`http://`, `https://`, `s3://`, `gs://`). The
pipeline streams it via `bam2bw` / `tangermeme.io` without downloading, and the
input check skips it. Credentialed buckets need the usual environment
variables (`AWS_*`, `GOOGLE_APPLICATION_CREDENTIALS`) set first.
