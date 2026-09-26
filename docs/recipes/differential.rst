Recipe: Differential / Conditional Analysis
===========================================

This recipe covers comparing two (or more) conditions — treated vs.
control, knockout vs. wildtype, time point A vs. B. The standard
pattern in Cherimoya is to train one model per condition, then
compare their predictions on a shared set of loci. This is simpler
to set up than a single multi-output model and gives cleaner
attribution and marginalization results.

If you have only a single condition, use the assay-specific recipe
(:doc:`chipseq_tf`, :doc:`atacseq`, or :doc:`dnaseq`) instead.


Inputs
------

* Reference genome FASTA.
* Per-condition signal BAMs (with replicates pooled or listed as
  separate ``signals`` files).
* Per-condition control BAMs (for ChIP-seq) or none (for ATAC/DNase).
* A motif database in MEME format.


Step 1: train one model per condition
--------------------------------------

Name the files after each condition and run one sweep over the
condition names. ``${name}`` in a value is replaced by each job's
``name``, so every job reads its own BAMs:

.. code-block:: bash

   cherimoya pipeline -m name=condA,condB sequences=hg38.fa \
       loci=null negatives=null \
       'signals=[${name}_rep1.bam,${name}_rep2.bam]' \
       'controls=[${name}_input.bam]' \
       motifs=JASPAR_2024.meme \
       hydra.sweep.dir=differential 'hydra.sweep.subdir=${name}'

``-m`` must come before or after the ``key=value`` overrides, not
between them. The two ``hydra.sweep`` keys put each job in
``differential/<name>/`` instead of the default
``multirun/<date>/<time>/<n>/``. The jobs run one after another; to
run them in parallel on a SLURM cluster, see :doc:`../cli`.

Each job produces a model checkpoint (``differential/condA/condA.torch``,
``differential/condB/condB.torch``), per-track bigWigs, attributions,
seqlets, and a TF-MoDISco report scoped to that condition's peaks.

Both jobs share one config apart from ``name``, so they use the same
``fit.training_chroms`` / ``fit.validation_chroms`` and the held-out
evaluation is comparable.


Step 2: build a shared locus set
--------------------------------

Decide what regions to compare across. The natural choices, in
increasing order of conservatism:

* **Union** of condition-A and condition-B peaks — catches gains and
  losses but includes weakly-supported regions.
* **Intersection** — only regions called as peaks in both
  conditions; conservative but misses pure gains/losses.
* **Union over a reference annotation** (e.g. promoters,
  GENCODE-defined TSSes) — gives a biologically interpretable set
  that doesn't depend on the peak calls.

A typical union with ``bedtools``:

.. code-block:: bash

   cat differential/condA/condA_peaks.narrowPeak \
       differential/condB/condB_peaks.narrowPeak | \
       sort -k1,1 -k2,2n | \
       bedtools merge -i - > shared_loci.bed


Step 3: predict from both models on the shared set
--------------------------------------------------

.. code-block:: python

   from cherimoya import Cherimoya
   from tangermeme.io import extract_loci
   from tangermeme.predict import predict

   model_A = Cherimoya.load("differential/condA/condA.torch", device="cuda")
   model_B = Cherimoya.load("differential/condB/condB.torch", device="cuda")
   model_A.eval(); model_B.eval()

   X, _ = extract_loci(
       sequences="hg38.fa",
       loci="shared_loci.bed",
       chroms=["chr8", "chr20"],     # the held-out chromosomes used during training
       in_window=2114, out_window=1000, max_jitter=0,
       ignore=list('QWERYUIOPSDFHJKLZXVBNM'),
       return_mask=True,
   )

   y_prof_A, y_counts_A = predict(model_A, X, batch_size=64, device="cuda")
   y_prof_B, y_counts_B = predict(model_B, X, batch_size=64, device="cuda")

   # Differences in log counts (predicted log fold change).
   delta_log_counts = y_counts_B - y_counts_A

Loci with large positive ``delta_log_counts`` are predicted to gain
signal in condition B; large negative values are predicted losses.


Step 4: identify differential motifs
------------------------------------

For motif-level differences, run marginalization on both models and
compare. Each model's pipeline already runs marginalization on its
own peaks; to compare directly, run marginalization on a
**shared** background:

.. code-block:: bash

   cherimoya marginalize model=differential/condA/condA.torch \
       sequences=hg38.fa motifs=JASPAR_2024.meme \
       'loci=[shared_loci.bed]' output_filename=condA_shared_marginalize/
   cherimoya marginalize model=differential/condB/condB.torch \
       sequences=hg38.fa motifs=JASPAR_2024.meme \
       'loci=[shared_loci.bed]' output_filename=condB_shared_marginalize/

The two commands differ only in ``model`` and ``output_filename``,
and share ``loci`` (the shared background) and ``motifs``. The
delta in per-motif marginalization scores between A and B is a
direct estimate of which motifs cause the condition-specific signal.


Caveats
-------

* **Held-out chromosomes must match.** A model trained with
  ``chr8``/``chr20`` as validation cannot be compared with one
  trained with ``chr1``/``chr12`` as validation on common loci —
  some of those loci were in the second model's training set. Use
  the same split for both models.
* **Replicate-to-replicate noise is the floor.** Before treating
  ``delta_log_counts`` as biological signal, compare to the
  technical-replicate baseline by training two models on independent
  replicates of the same condition and computing the same delta.
  Biological deltas should be larger than the replicate baseline.
* **The two models share no parameters.** Each model is fit
  independently and the comparison happens only at prediction time.
  Multi-output single-model training is possible by passing
  ``'signals=[condA.bw,condB.bw]'`` to one pipeline (Cherimoya treats each
  signal as a separate output track), but in practice per-condition
  models give cleaner attribution and motif results.
