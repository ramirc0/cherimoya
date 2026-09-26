# Assay-specific settings

The assay determines a handful of preprocessing options that change the biology
of what gets modeled: strandedness, fragments vs reads, paired-end handling, and
read shifts. None can be inferred from a filename, so **confirm the assay and
each setting with the user** rather than assuming. The tables below are starting
points to *propose and confirm*, not values to apply silently.

All of them are keys in the pipeline's `preprocessing` section. Set them as
overrides (`preprocessing.unstranded=true`) or in the YAML file:

```yaml
preprocessing:
  unstranded: true
  pos_shift: 4
  neg_shift: -4
```

## The knobs

| Key | Meaning |
|---|---|
| `preprocessing.unstranded` | Produce one unstranded track instead of a `(+, -)` pair. |
| `preprocessing.fragments` | Input is fragment files, not aligned reads. |
| `preprocessing.paired_end` | Input is paired-end; affects MACS3 format (`BAMPE`). |
| `preprocessing.pos_shift` | Shift `+` strand reads by N bp. |
| `preprocessing.neg_shift` | Shift `-` strand reads by N bp. |
| `preprocessing.scale_factor` | Multiply raw counts by X (default 1.0 = no scaling). |
| `preprocessing.callpeaks_gsize` | MACS3 effective genome size: `hs` human, `mm` mouse. |
| `preprocessing.callpeaks_q` | MACS3 q-value cutoff (default 0.05). Loosen to 0.1 for low-yield; tighten to 0.01 for high-confidence. |

Every shift defaults to **0** and strandedness defaults to **stranded**
(`unstranded: false`), so doing nothing gives a stranded, unshifted, read-based
run. That is *wrong* for ATAC and DNase, which is why confirming the assay
matters.

## Reads vs. fragments (`fragments` and `paired_end`)

For ATAC-seq and DNase-seq especially, the biggest thing to get right is whether
the input holds **aligned reads** or **fragments**; it decides `fragments` and
`paired_end`. These are *not* interchangeable and depend only on the file, not
the assay:

| Input | Typical file | `fragments` | `paired_end` |
|---|---|---|---|
| **Fragment file** | `.tsv`, `.tsv.gz` (also `.bed`/`.bed.gz`) | **true** | **false** |
| **Paired-end reads** | `.bam` (paired-end ATAC) | false | **true** |
| **Single-end reads** | `.bam` (typical DNase) | false | false |

Why (from the pipeline source):
- **`fragments`** is what routes a fragment file correctly: MACS3 uses `FRAG`
  format and `bam2bw` parses fragment intervals rather than reads.
- **`paired_end`** exists *only* to switch MACS3 to `BAMPE` for a paired-end
  **BAM of reads**. A fragment file is already `FRAG`, so **`paired_end` has no
  effect on fragments**. Don't set it there.

So a fragment file gets `fragments=true` and *not* `paired_end`; a paired-end
read BAM gets `paired_end=true` and *not* `fragments`. If unsure which a user
has, ask, or infer from the extension and confirm (see the key/filetype check
in `references/input-files.md`).

## Starting points by assay (confirm before applying)

### ATAC-seq (unstranded, +4 / -4 Tn5 shift)
Unstranded, Tn5 insertion conventionally corrected with a **+4 / -4** shift. The
rest depends on reads vs. fragments:

- **Paired-end read BAM** (`atac.bam`): `preprocessing.unstranded=true
  preprocessing.pos_shift=4 preprocessing.neg_shift=-4
  preprocessing.paired_end=true`
- **Fragment file** (`atac.fragments.tsv.gz`): `preprocessing.unstranded=true
  preprocessing.pos_shift=4 preprocessing.neg_shift=-4
  preprocessing.fragments=true` (`fragments`, not `paired_end`; see the table.)
- **Ask whether the data is already shifted.** Many pipelines
  (10x/CellRanger/ArchR-derived fragments) pre-apply the Tn5 shift; applying it
  twice is wrong. If already shifted, drop the shifts (keep `unstranded` and the
  reads/fragments key), or, if the upstream shift used the older +4/-5
  convention, set the shifts to the relative offset rather than zero.

### DNase-seq (unstranded, usually single-end, no shift)
Typically a single-end read BAM modeled as one unstranded track:
`preprocessing.unstranded=true`.
- No `paired_end` (single-end), no `fragments` (aligned reads, not fragments).
- No shift by default; some protocols apply a small `+1 / 0` cut-site shift
  (`preprocessing.pos_shift=1 preprocessing.neg_shift=0`) for footprint work.
  Confirm.
- For a paired-end DNase read BAM, add `preprocessing.paired_end=true`; for a
  stranded variant, leave `unstranded` at `false`.

### ChIP-seq (TF or histone)
- **TF ChIP-seq is stranded by default** (`+` and `-` coverage modeled as two
  tracks), single-end, no shift, **with an input control**:
  `'signals=[chip.bam]' 'controls=[input.bam]'`.
  Leave `unstranded` alone; the default (`false`) is what you want.
- `signals` is the ChIP/IP signal; `controls` the unenriched input control.
  Confirm the user has the matched input. A control-trained model must be used
  with controls thereafter.
- If **paired-end**, add `preprocessing.paired_end=true`. Histone ChIP is
  *sometimes* modeled unstranded (`preprocessing.unstranded=true`), a
  confirm-with-user choice, not a default.

### Stranded assays (many TF profiling / initiation assays, PRO-seq, CAGE)
- Leave strandedness on (`unstranded: false`); the pipeline emits a `(+, -)`
  pair as one stranded group.
- For pre-made bigWigs, remember the nested grouping
  (`'signals=[[plus.bw,minus.bw]]'`) from `references/input-files.md`.

### Non-human genome
- Set `preprocessing.callpeaks_gsize` to `mm` for mouse (or a numeric effective
  size), **and** replace the hg38 `fit.training_chroms` /
  `fit.validation_chroms` with names/splits valid for that genome. The
  chromosome default is the most commonly missed one for non-human data.

## When you're unsure

If the user can't say whether the data is shifted, stranded, or paired-end,
**ask one question** rather than pick a default. An unnecessary read shift or
wrong strandedness degrades the model with no error message. A safe fallback for
a totally unknown coverage track is unstranded and unshifted
(`preprocessing.unstranded=true`), which at least won't double-apply a
correction.

Full recipes with worked commands: the ChIP-seq, ATAC-seq, and DNase-seq pages
under <https://cherimoya.readthedocs.io/en/latest/> (recipes section).
