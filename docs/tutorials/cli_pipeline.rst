CLI Pipeline Walkthrough
========================

This page is the user's guide to the end-to-end Cherimoya CLI: how to
take raw BAM/BED files and a reference genome through peak calling,
training, attribution, seqlet calling, motif discovery, and motif
marginalization in a single reproducible run.

.. image:: ../../imgs/pipeline.png
   :align: center
   :width: 70%
   :alt: Cherimoya end-to-end pipeline

|

For every config key and its default, see :doc:`../cli`. For assay-specific recipes
(TF ChIP-seq, ATAC-seq, DNase-seq) see the recipe pages.


Prerequisites
-------------

You will need:

- Cherimoya installed (see :doc:`../installation`).
- A reference genome FASTA file (e.g. ``hg38.fa``).
- One or more signal files (BAM/SAM, fragment BED/TSV, or bigWig).
- A motif database in MEME format (optional, used by TF-MoDISco
  reports and marginalization).
- Optional: a BED of peak coordinates. If you don't provide one, the
  pipeline calls peaks with MACS3.
- Optional: BED of GC-matched negative regions. If you don't provide
  one, the pipeline samples them automatically.

All file inputs can be remote URLs (``http://``, ``https://``,
``s3://``, ``gs://``). ``bam2bw`` streams the data directly without
downloading the file first; the validation step before the run only
checks paths that look local.


Subcommands at a glance
-----------------------

``cherimoya pipeline`` runs the whole thing. Every other subcommand
(``fit``, ``evaluate``, ``attribute``, ``seqlets``, ``marginalize``,
``negatives``) is one pipeline stage and can be run on its own. The
pipeline writes a config file for each stage it runs, so any stage can
be rerun alone. See :doc:`../cli` for the full reference.


Configuring a run
-----------------

Each command reads its config from an optional YAML file (``-p``) and
``key=value`` overrides on the command line; anything not given takes
its default. The pipeline needs five keys: ``name``, ``sequences``,
``loci``, ``negatives`` and ``signals``. Set ``loci`` or
``negatives`` to ``null`` to have the pipeline produce them.

For stranded ChIP-seq with input controls:

.. code-block:: bash

   cherimoya pipeline name=my_experiment sequences=hg38.fa \
       'loci=[peaks.narrowPeak]' negatives=null \
       'signals=[input1.bam,input2.bam]' \
       'controls=[control1.bam,control2.bam]' \
       motifs=JASPAR_2024.meme

For unstranded paired-end ATAC-seq with the standard +4 / -4 fragment
shift:

.. code-block:: bash

   cherimoya pipeline name=atac_experiment sequences=hg38.fa \
       'loci=[peaks.narrowPeak]' negatives=null \
       'signals=[fragments.bam]' motifs=JASPAR_2024.meme \
       preprocessing.pos_shift=4 preprocessing.neg_shift=-4 \
       preprocessing.unstranded=true preprocessing.fragments=true \
       preprocessing.paired_end=true

Lists must be quoted so the shell leaves the brackets alone. Dotted
keys such as ``preprocessing.unstranded`` or ``fit.n_filters`` set one
step's section.

Keeping the config in a file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a run you will repeat or tweak, write the config to a file first.
``--cfg job`` prints the full config, every key with its default, and
runs nothing:

.. code-block:: bash

   cherimoya pipeline name=my_experiment sequences=hg38.fa \
       'loci=[peaks.narrowPeak]' negatives=null \
       'signals=[input1.bam,input2.bam]' \
       'controls=[control1.bam,control2.bam]' \
       motifs=JASPAR_2024.meme --cfg job > pipeline.yaml

Edit ``pipeline.yaml`` to change any default: model width, training
and validation chromosomes, seqlet p-value threshold, MoDISco
settings. Values such as ``${name}`` link a step's key to the
top-level key, so changing ``name`` at the top renames every output.
To train on several GPUs, set ``fit.devices`` to their number.
``fit.batch_size`` is then the global batch, split evenly across them
(see :ref:`training on several devices <cli-several-devices>`).
Then run it, with or without further overrides:

.. code-block:: bash

   cherimoya pipeline -p pipeline.yaml
   cherimoya pipeline -p pipeline.yaml fit.n_filters=64

A typo in the file or an override is an error, and so is a required key
left as ``???``. Local input files are checked before any work starts,
and every missing one is listed. Remote URLs are not checked.

What the pipeline does
~~~~~~~~~~~~~~~~~~~~~~

In order:

1. **MACS3 peak calling** (skipped if ``loci`` is set).
2. **bam2bw conversion** to bigWig (skipped if signals are already
   bigWigs).
3. **GC-matched negative sampling** (skipped if ``negatives`` is set).
4. **Model training** (skipped if ``model`` is set). Writes
   ``{name}.torch`` (best checkpoint by validation count Pearson) and
   ``{name}.final.torch`` (EMA weights at end of training), plus
   ``{name}.log``, then evaluates the best checkpoint on the
   validation and test chromosomes into
   ``{name}.validation.performance.tsv`` and
   ``{name}.test.performance.tsv``.
5. **Attribution** via DeepLIFT/SHAP (or saturation mutagenesis, with
   ``attribute.algorithm``), kept over the central 400 bp of each
   example, saved as ``{name}.attributions.{ohe,attr}.npz`` and
   ``{name}.attributions.idxs.npy``.
6. **Seqlet identification** with TF-MoDISco-style recursive seqlet
   calling on the (attribution × one-hot) signal, written to
   ``{name}.seqlets.bed``.
7. **tomtom-lite seqlet annotation** against the motif database, if
   ``motifs`` was provided; results in
   ``{name}.seqlets_annotated.bed`` and a counts table in
   ``{name}.motif_seqlet_count.tsv``.
8. **TF-MoDISco motif discovery** and HTML report; results in
   ``{name}_modisco_results.h5`` and ``{name}_modisco/``.
9. **Marginalization**, if ``motifs`` was provided. Measures the
   predicted effect of inserting each motif into background sequences
   drawn from ``negatives``. Output in ``{name}_marginalize/``.

Before each of steps 3, 4, 5, 6 and 9, the pipeline saves that step's
config as ``{name}.<command>.yaml``. With ``dry_run=true`` it writes
these files and runs nothing, which is a cheap way to check a config.


Running individual steps
------------------------

Each stage has its own subcommand. The files the pipeline saved are
complete configs for them:

.. code-block:: bash

   cherimoya fit -p my_experiment.fit.yaml
   cherimoya evaluate -p my_experiment.test.evaluate.yaml
   cherimoya attribute -p my_experiment.attribute.yaml
   cherimoya seqlets -p my_experiment.seqlets.yaml
   cherimoya marginalize -p my_experiment.marginalize.yaml

Overrides work here too, so rerunning one stage with a change needs no
editing:

.. code-block:: bash

   cherimoya seqlets -p my_experiment.seqlets.yaml threshold=0.001 \
       output_filename=my_experiment.strict.seqlets.bed

``cherimoya <command> --help`` prints a command's keys and defaults.
Every command but ``negatives`` also takes ``skip=true`` to no-op it.
In a pipeline, ``attribute.skip=true`` skips one step and ``skip=true``
skips them all.


Training many models
--------------------

``-m`` runs one job per value, or per combination of values:

.. code-block:: bash

   cherimoya fit -p my_experiment.fit.yaml -m random_state=0,1,2

Each job runs in its own ``multirun/<date>/<time>/<n>/`` directory, so
the three models don't overwrite each other. Relative input paths
still resolve against the directory you started from. To send each job
to a SLURM cluster, see :doc:`../cli`.


About bam2bw
------------

The pipeline does not call BAM/SAM/fragment files directly into the
training step — it converts them to bigWig first using the
``bam2bw`` tool, which is a hard dependency. ``bam2bw`` streams the
input (local or remote URL), counts reads or fragments into per-base
coverage, optionally applies a ± shift (used for Tn5 / DNase
corrections), and emits one bigWig (unstranded) or two bigWigs
(``+`` / ``-`` stranded).

This conversion is what enables remote URLs as inputs: ``bam2bw``
fetches reads via byte-range requests rather than downloading the
whole file. The resulting bigWigs are written into the working
directory and re-used by every downstream stage.

If your signals are already bigWigs, the pipeline skips this step
automatically. To use ``bam2bw`` standalone (outside the pipeline),
invoke it directly — see its own documentation.


Calling negatives independently
-------------------------------

``cherimoya pipeline`` calls negatives for you when ``negatives`` is
``null``. If you want to sample GC-matched
negatives without running the full pipeline (e.g. you're going to
train a non-Cherimoya model on the same regions), use the
``negatives`` subcommand:

.. code-block:: bash

   cherimoya negatives peaks=peaks.narrowPeak fasta=hg38.fa \
       bigwig=signal.bw output=negatives.bed \
       bin_width=0.02 max_n_perc=0.1 beta=0.5

The output is a 3-column BED of regions matched by GC content to the
input peaks, with at most ``max_n_perc`` fraction of ``N`` bases
and (optionally) signal below ``beta × min(peak_counts)``.


Running on a non-hg38 reference
-------------------------------

The defaults assume hg38. To run on a different reference (mouse mm10,
non-human, or a different hg version), override four pipeline keys:

* ``fit.training_chroms``: chromosomes used to train.
  Replace with the appropriate list for your reference (e.g. mm10:
  ``[chr1, chr2, …, chr19, chrX, chrY]`` minus the
  validation and test chromosomes you choose).
* ``fit.validation_chroms``: held-out chromosomes for
  validation, which choose the checkpoint. Two chromosomes is enough.
* ``fit.test_chroms``: held-out chromosomes evaluated once after
  training, or ``null`` for no test evaluation. ``fit`` refuses to
  start if two of the three lists share a chromosome.
* ``preprocessing.callpeaks_gsize``: MACS3 effective
  genome size. Use ``"mm"`` for mouse, ``"ce"`` for *C. elegans*,
  ``"dm"`` for fly, or a numeric value (e.g. ``"2.7e9"`` for hg38) for
  any other organism.

For non-chromosome reference contigs (scaffolds, alternate
haplotypes, viral integrations), exclude them by listing them
explicitly in the training/validation chromosome lists, or by passing
an ``exclusion_lists`` BED.


Outputs
-------

A successful pipeline run leaves the following in the working
directory (with ``{name}`` from the ``name`` key):

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - File
     - Contents
   * - ``{name}.torch``
     - Best-by-validation-count-Pearson checkpoint (config + state_dict).
   * - ``{name}.final.torch``
     - Final EMA-applied checkpoint at end of training.
   * - ``{name}.log``
     - Per-epoch training and validation metrics (TSV). Same
       columns regardless of how many signal groups the model has.
   * - ``{name}.detailed.log``
     - Same as ``{name}.log`` plus one ``ProfilePearson_g{i}``,
       ``CountPearson_g{i}``, ``AUROC_g{i}`` and ``AUPRC_g{i}``
       column per signal group, for offline per-modality analysis.
       Never printed to stdout.
   * - ``{name}.validation.performance.tsv`` / ``{name}.test.performance.tsv``
     - Metrics of the best checkpoint on the validation chromosomes
       (which chose it) and on the test chromosomes (which did not).
       Each has one TSV row per signal
       group, in ``signal_groups`` order; single-group models
       write exactly one row. Seven columns computed on the peaks,
       then five computed with the negatives: the count Pearson,
       Spearman and MSE over peaks and negatives, and the AUROC and
       AUPRC.
   * - ``{name}.+.bw`` / ``{name}.-.bw``
     - bigWigs produced by ``bam2bw`` for the stranded signal.
   * - ``{name}.bw``
     - bigWig produced by ``bam2bw`` for unstranded signal.
   * - ``{name}.control.{+,-}.bw``
     - bigWigs produced by ``bam2bw`` for stranded controls.
   * - ``{name}_peaks.narrowPeak``
     - Peaks called by MACS3 (when ``loci`` not provided).
   * - ``{name}.negatives.bed``
     - GC-matched negative regions (when ``negatives`` not provided).
   * - ``{name}.attributions.ohe.npz``
     - One-hot encoded sequences over the central 400 bp window.
   * - ``{name}.attributions.attr.npz``
     - Hypothetical importance scores (DeepLIFT/SHAP by default).
   * - ``{name}.attributions.idxs.npy``
     - Boolean mask into the original loci list selecting examples
       that had no Ns over the window.
   * - ``{name}.seqlets.bed``
     - Recursive seqlets in genome coordinates.
   * - ``{name}.seqlets_annotated.bed``
     - Seqlets with closest-motif annotation from tomtom-lite.
   * - ``{name}.motif_seqlet_count.tsv``
     - Count of seqlets matched per motif.
   * - ``{name}_modisco_results.h5``
     - TF-MoDISco pattern HDF5.
   * - ``{name}_modisco/``
     - TF-MoDISco HTML report.
   * - ``{name}_marginalize/``
     - Motif marginalization report (HTML with PNG figures).
   * - ``{name}.{negatives,fit,attribute,seqlets,marginalize}.yaml``
     - Per-step configs, each a ``-p`` file for rerunning that step.
   * - ``{name}.{validation,test}.evaluate.yaml``
     - The evaluate configs ``fit`` writes and runs.
   * - ``.hydra/pipeline/``
     - The composed pipeline config (``config.yaml``) and the
       command-line overrides (``overrides.yaml``).
