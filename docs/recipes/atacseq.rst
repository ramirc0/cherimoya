Recipe: ATAC-seq
================

This recipe trains an unstranded Cherimoya model on paired-end
ATAC-seq, applying the standard +4 / −4 fragment shift to match the
Tn5 insertion offsets. ATAC-seq is typically modeled as a single
unstranded output track.


Inputs
------

* Reference genome FASTA (e.g. ``hg38.fa``).
* A BAM file of aligned ATAC-seq paired-end reads, **or** a BED/TSV
  fragment file.
* A motif database in MEME format.
* Optional: a BED of peak coordinates. If not provided, MACS3 will
  call peaks.
* Optional: a BED of negative coordinates. If not provided, we will
  automatically identify GC-matched negatives.


About the +4 / −4 shift
-----------------------

Tn5 transposes 9 bp apart and the standard correction many tools
apply is +4 / −5. This recipe uses **+4 / −4** because the model treats
the two end events symmetrically; the pipeline's own ``pos_shift`` and
``neg_shift`` default to 0, so the shift has to be passed. 
Apply the shift here, not upstream, and **don't double-shift** — 
if your BAM is already shifted, set the shifts to zero, or to the relative
offset if converting from +4 / -5. This idea was introduced with the
ChromBPNet model.


Run the pipeline
----------------

For a paired-end BAM:

.. code-block:: bash

   cherimoya pipeline name=atac_experiment sequences=hg38.fa \
       loci=null negatives=null 'signals=[atac.bam]' \
       motifs=JASPAR_2024.meme \
       preprocessing.pos_shift=4 preprocessing.neg_shift=-4 \
       preprocessing.unstranded=true preprocessing.paired_end=true

Key by key:

* ``preprocessing.pos_shift=4`` / ``preprocessing.neg_shift=-4``: Tn5
  shift on plus and minus ends.
* ``preprocessing.unstranded=true``: unstranded output (single signal
  track).
* ``preprocessing.paired_end=true``: paired-end. Causes MACS3 to use
  ``BAMPE`` file format rather than ``BAM``.

Pass ``'loci=[peaks.bed]'`` or ``'negatives=[negatives.bed]'`` instead
of ``null`` if you have them.

If your input is already a fragments TSV/BED (e.g. from
``snap-atac``, ``CellRanger``, or a custom ``samtools`` pipeline),
pass the fragment file in ``signals``, add
``preprocessing.fragments=true`` to indicate that the file is a
fragment file, and drop ``preprocessing.paired_end=true``; ``bam2bw``
detects the file extension and handles the fragment file format.

The steps mirror the ChIP-seq recipe, with these differences:

* MACS3 runs without a control file and with format ``BAMPE`` (or
  ``FRAG`` for fragment-file input).
* ``bam2bw`` is invoked with the ``-u`` (unstranded) and ``-ps 4
  -ns -4`` flags, plus ``-f`` for fragment-file input, producing a
  single ``atac_experiment.bw`` rather than ``+.bw`` / ``-.bw`` pair.
* The trained Cherimoya model has ``signal_groups=[1]`` (one
  unstranded group) and ``n_control_tracks=0``.
* Attribution, seqlet calling, TF-MoDISco, and marginalization run
  the same way they do for ChIP-seq.


ATAC-seq peak counts can be noisier
-----------------------------------

ATAC-seq peaks span a wider range of summit heights than TF ChIP-seq.
The ``PeakGenerator`` filter drops peaks above ``1.2 ×`` the 99th
percentile of summed counts, which removes the extreme outliers. One
parameter is worth checking after a first training run:

* ``fit.max_jitter``: the default of 500 bp randomly
  shifts peak centers each epoch, which improves training-set
  diversity; ATAC-seq peaks are typically wide enough to tolerate
  this. The loader extracts ``in_window + 2 * max_jitter`` bp around
  each peak, so a shift never cuts into the flank between
  ``in_window`` and ``out_window``.


Outputs
-------

See the "Outputs" table in :doc:`../tutorials/cli_pipeline`. ATAC-seq
runs produce one unstranded bigWig (``atac_experiment.bw``) rather
than a stranded pair, and the model has a single profile track. The
HTML reports under ``atac_experiment_modisco/`` and
``atac_experiment_marginalize/`` are the same format as ChIP-seq.
