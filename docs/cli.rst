CLI Reference
=============

Reference for every ``cherimoya`` subcommand and every config key. The
defaults come from the schemas in ``cherimoya_cli/config.py``; update
these tables when that file changes.

For a walkthrough of how the pieces fit together see
:doc:`tutorials/cli_pipeline`.


Running a command
-----------------

.. code-block:: text

   cherimoya <command> [-p FILE] [key=value ...] [-m] [Hydra flags]
   cherimoya install-skill [-d DIR] [--symlink] [-f]
   cherimoya --version

``<command>`` is one of ``pipeline``, ``negatives``, ``fit``,
``evaluate``, ``attribute``, ``seqlets`` or ``marginalize``. These are
configured with `Hydra <https://hydra.cc>`_: each command has a typed
schema in ``cherimoya_cli/config.py``, so every key has a type and a
default, and an unknown key is an error. ``cherimoya <command> --help``
prints the command's full config.

A config comes from three layers, each overriding the one before:

1. The schema defaults, listed in the tables below.
2. An optional YAML file passed with ``-p FILE``.
3. ``key=value`` overrides on the command line.

.. code-block:: bash

   cherimoya fit -p run.yaml n_filters=64 max_epochs=10

The ``-p`` file is a plain YAML mapping of keys to values. It needs no
Hydra ``defaults:`` header, and it is checked against the schema like
the overrides are, so a typo in either place fails before any work
starts:

.. code-block:: text

   Key 'n_filter' not in 'FitConfig'

JSON parameter files from earlier versions of Cherimoya no longer load.
Write the same keys as YAML, dropping the ``_parameters`` suffix from
the pipeline's step sections (``fit_parameters`` becomes ``fit``).

Override syntax
~~~~~~~~~~~~~~~

* Nested keys are dotted: ``fit.n_filters=64`` sets the pipeline's fit
  step, ``preprocessing.unstranded=true`` its preprocessing.
* Lists use brackets and must be quoted, or the shell expands the
  brackets: ``'signals=[a.bw,b.bw]'``,
  ``'signals=[[ctcf.+.bw,ctcf.-.bw]]'``,
  ``'validation_chroms=[chr8,chr20]'``.
* ``null`` sets a key to null: ``negatives=null``.
* Paths may be remote URLs such as ``'loci=[s3://bucket/peaks.bed.gz]'``.
* ``${key}`` in a value refers to another key. Quote it, since the shell
  would expand it: ``'signals=[${name}.bw]'``.
* Hydra reads the overrides as one unbroken run, so flags such as
  ``-m`` and ``--cfg`` go before or after all of them, not between
  them. ``-p FILE`` may go anywhere.

Required keys and ``null``
~~~~~~~~~~~~~~~~~~~~~~~~~~

A key shown as ``???`` in ``--help`` or in a template has no default and
must be given. The tables below mark these as *required*. If any are
missing, the command lists all of them in one error and stops:

.. code-block:: text

   Must provide a value for: loci, negatives, sequences, signals. Set a key to null if an earlier pipeline step produces it.

Some required keys accept an explicit ``null``. In a ``pipeline``
config, ``null`` means "an earlier step produces this": ``loci: null``
calls peaks with MACS3, and ``negatives: null`` samples GC-matched
negatives. Leaving the key out is not the same as ``null``; it is an
error.

Input files
~~~~~~~~~~~

Before a command runs, every local input path is made absolute
(relative paths resolve against the directory the command was started
from) and checked. All missing files are reported together:

.. code-block:: text

   FileNotFoundError: The following inputs are missing:
     - sequences: /data/run/hg38.fa
     - signals: /data/run/a.bw

Remote paths (``http://``, ``https://``, ``s3://``, ``gs://``) are not
checked. ``bam2bw`` and ``tangermeme.io`` stream them.

Writing a config file
~~~~~~~~~~~~~~~~~~~~~

``--cfg job`` prints the composed config as YAML and runs nothing. It
is the way to start a config file:

.. code-block:: bash

   cherimoya pipeline name=my_experiment sequences=hg38.fa \
       'loci=[peaks.narrowPeak]' negatives=null 'signals=[input.bam]' \
       --cfg job > run.yaml
   cherimoya pipeline -p run.yaml

Keys you did not give print as ``???``; fill them in before running.
In a pipeline template, the step sections keep their links to the
top-level keys (``name: ${name}``, ``loci: ${loci}``), so editing
``name`` or ``in_window`` at the top still reaches every step. Hydra
does not accept ``--cfg`` together with ``-m``.

What a run writes
~~~~~~~~~~~~~~~~~

A command writes its outputs into the directory it was started from.
Hydra also saves the composed config and the command-line overrides in
``.hydra/<command>/`` (``config.yaml``, ``overrides.yaml`` and
``hydra.yaml``). ``config.yaml`` holds the config before the command
runs, so values a run fills in, such as a drawn ``random_state``, are
not in it.

Two commands write configs for their steps, each a valid ``-p`` file
for rerunning that step alone:

* ``fit`` writes ``<name>.validation.evaluate.yaml`` and
  ``<name>.test.evaluate.yaml``, one before each evaluation.
* ``pipeline`` writes ``<name>.<command>.yaml`` for each step it runs:
  ``negatives``, ``fit``, ``attribute``, ``seqlets`` and
  ``marginalize``.

Sweeps and SLURM
~~~~~~~~~~~~~~~~

``-m`` (``--multirun``) runs one job per combination of comma-separated
values:

.. code-block:: bash

   cherimoya fit -p run.yaml -m random_state=0,1,2
   cherimoya fit -p run.yaml -m n_filters=64,128 random_state=0,1

Each job runs in its own directory, ``multirun/<date>/<time>/<n>/``,
with its own outputs and ``.hydra/<command>/``, so jobs never overwrite
each other. The sweep directory also holds ``multirun.yaml``. Input
paths resolve against the directory the sweep was started from, so
relative inputs still work inside a job directory.

By default the jobs run one after another in the current process. To
submit each job to a SLURM cluster, install the ``slurm`` extra, which
adds the `submitit launcher
<https://hydra.cc/docs/plugins/submitit_launcher/>`_:

.. code-block:: bash

   pip install "cherimoya[slurm]"
   cherimoya fit -p run.yaml -m random_state=0,1,2 \
       hydra/launcher=submitit_slurm hydra.launcher.partition=gpu \
       hydra.launcher.gpus_per_node=1 hydra.launcher.timeout_min=240

``hydra/launcher=submitit_local`` runs the jobs as local subprocesses
instead. Both launchers keep their logs in ``.submitit/`` inside the
sweep directory. ``cherimoya fit ... hydra/launcher=submitit_slurm
--cfg hydra`` prints every launcher setting.

Other conventions
~~~~~~~~~~~~~~~~~

* ``skip=true`` makes ``fit``, ``evaluate``, ``attribute``, ``seqlets``
  or ``marginalize`` a no-op. In ``pipeline``, a top-level ``skip=true``
  no-ops the whole pipeline and ``annotation.skip=true`` skips the
  seqlet annotation; the MoDISco steps have no ``skip``. Inputs are
  still checked first, so a missing file fails even when skipped.
* ``pipeline`` takes ``dry_run=true``, which writes the per-step YAML
  files and runs nothing.
* ``signals`` and ``controls`` take a flat list of files, each its own
  one-channel (unstranded) group, or a grouped list whose entries are
  each a file or a list of files (a multi-channel group such as a
  stranded ``(+, -)`` pair). ``[atac.bw, [ctcf.+.bw, ctcf.-.bw]]``
  declares one unstranded ATAC group and one stranded CTCF group. The
  grouping decides how reverse-complement augmentation permutes
  channels and how many count predictions the model makes. A stranded
  pair must use the nested form ``[[plus.bw, minus.bw]]``; a flat
  ``[plus.bw, minus.bw]`` is two unstranded groups. See
  :doc:`multi_task`.
* ``loci``, ``negatives`` and ``exclusion_lists`` take a list of BED
  files. ``pipeline`` also takes a single path for ``loci`` and
  ``negatives``.

Some keys are declared but not read by any command. Setting them has no
effect; the tables below say so where it applies.


cherimoya fit
-------------

Train a model, then evaluate the best checkpoint twice: on the
``validation_chroms`` and on the ``test_chroms``. Before each
evaluation, ``fit`` writes ``<name>.validation.evaluate.yaml`` or
``<name>.test.evaluate.yaml`` and runs ``evaluate`` with it, writing
``<name>.validation.performance.tsv`` or
``<name>.test.performance.tsv``. The evaluations use fit's
``sequences``, ``loci``, ``negatives``, ``signals``, ``controls``,
``exclusion_lists``, ``batch_size``, ``in_window``, ``out_window``,
``reverse_complement_average``, ``summits``, ``compile``,
``compile_mode``, ``dtype``, ``device`` and ``verbose``. Their other
keys take the evaluate defaults. The validation numbers come from the
chromosomes that chose the checkpoint; the test numbers do not.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Key
     - Default
     - Description
   * - ``name``
     - required
     - Prefix of the output files. ``null`` uses the model's own name,
       ``cherimoya.<n_filters>.<n_layers>``.
   * - ``sequences``
     - required
     - Reference genome FASTA.
   * - ``loci``
     - required
     - BED file(s) of peaks.
   * - ``negatives``
     - required
     - BED file(s) of GC-matched negatives. Those on the validation
       chromosomes also join validation (see below). ``null`` trains on
       the peaks alone, which needs ``negative_ratio=0``.
   * - ``signals``
     - required
     - Signal bigWigs, flat or grouped (see above).
   * - ``controls``
     - ``null``
     - Control bigWigs. Same grouping rule as ``signals``.
   * - ``exclusion_lists``
     - ``null``
     - BED file(s) of regions to exclude.
   * - ``training_chroms``
     - hg38 (chr2, chr4, chr5, chr7, chr9 to chr19, chr21, chr22, chrX,
       chrY)
     - Chromosomes used for training.
   * - ``validation_chroms``
     - ``[chr8, chr20]``
     - Held-out chromosomes for validation. They choose the checkpoint
       and drive early stopping.
   * - ``test_chroms``
     - ``[chr1, chr3, chr6]``
     - Held-out chromosomes evaluated once after training, for an
       estimate that took no part in choosing the checkpoint. ``null``
       skips the test evaluation. ``fit`` refuses to start if any two
       of the three lists share a chromosome.
   * - ``n_filters``
     - 128
     - Backbone channel width.
   * - ``n_layers``
     - 9
     - Number of Cheri Blocks.
   * - ``expansion``
     - 2
     - MLP expansion factor inside each Cheri Block.
   * - ``residual_scale``
     - 0.15
     - Fixed residual scalar.
   * - ``batch_size``
     - 64
     - Global training batch size, split evenly across ``devices``, so
       it must be divisible by ``devices``. Training raises an error if
       the training set holds fewer examples than one batch. The
       evaluations after training use it too.
   * - ``in_window`` / ``out_window``
     - 2114 / 1000
     - Input and output window sizes (bp).
   * - ``max_jitter``
     - 500
     - Maximum jitter (bp) for peak centers at training time: each
       epoch shifts every peak window by a whole number of bp from
       ``-max_jitter`` to ``+max_jitter``.
   * - ``reverse_complement``
     - ``true``
     - Augment training with reverse complements.
   * - ``reverse_complement_average``
     - ``false``
     - Not read by training. Passed to the evaluations after training,
       where it averages predictions over both strands.
   * - ``summits``
     - ``false``
     - Center loci on the narrowPeak summit column.
   * - ``max_epochs``
     - 20
     - Maximum training epochs. Raised at run time when it would buy
       fewer than ``min_total_steps`` optimizer steps.
   * - ``min_total_steps``
     - 20000
     - Minimum optimizer steps for the run. An epoch is one pass over the
       peaks, so ``max_epochs`` alone buys a step count proportional to
       how many peaks an experiment has; this raises ``max_epochs`` until
       the run reaches this many steps. The learning rate schedules are
       laid out over the raised value, so they stretch with it. ``null``
       disables the floor.
   * - ``loss_weights``
     - ``null``
     - Fixed ``[w0, w1]`` for the profile and count terms, replacing the
       learned Kendall weights ``lw0`` / ``lw1``. When set, the profile
       loss is divided by each signal group's own batch-mean read depth
       first, and the ``lw_*`` optimizer becomes inert.
       ``[1.333, 0.274]`` reproduces the operating point the learned
       weights reach. ``null`` keeps the Kendall weights. Under
       ``verbose``, the two weights are printed in place of the
       ``lw_*`` optimizer's hyperparameters.
   * - ``muon_lr``
     - 0.025
     - Muon learning rate.
   * - ``muon_wd``
     - 0.03
     - Muon weight decay.
   * - ``adam_lr``
     - 0.001
     - AdamW learning rate.
   * - ``adam_wd``
     - 0.0
     - AdamW weight decay.
   * - ``lw_lr``
     - 0.001
     - SGD learning rate for the Kendall uncertainty weights
       (``lw0``, ``lw1``).
   * - ``lw_wd``
     - 0.0
     - SGD weight decay for the Kendall uncertainty weights.
   * - ``lw_momentum``
     - 0.9
     - SGD momentum for the Kendall uncertainty weights.
   * - ``n_warmup_epochs``
     - 2
     - Number of epochs over which the LR is linearly warmed up from
       1% of its target before cosine decay begins.
   * - ``negative_ratio``
     - 0.25
     - Negatives per peak per epoch.
   * - ``num_workers``
     - 1
     - Data-loading workers per device.
   * - ``early_stopping``
     - ``null``
     - Stop after N consecutive epochs with no validation count
       Pearson improvement. ``null`` trains the full ``max_epochs``.
   * - ``random_state``
     - 0
     - Seeds the model's initialization and the sampler's draw order.
       See :ref:`what a seed fixes <reproducibility>`. ``null`` draws a
       seed and prints ``Drew random_state=N; set random_state=N to
       repeat this run.`` A standalone ``fit`` records the drawn seed
       only in that line; the pipeline also saves it in
       ``<name>.fit.yaml``.
   * - ``compile`` / ``compile_mode``
     - ``true`` / ``"max-autotune"``
     - Applied to the training model and to both evaluations, as for
       ``evaluate``.
   * - ``dtype``
     - ``"float32"``
     - Training precision, mapped to Lightning's ``"32-true"``,
       ``"bf16-mixed"`` (``"bfloat16"``) or ``"16-mixed"``
       (``"float16"``). The two half precisions run the forward pass
       under autocast, and ``"float16"`` also scales the loss.
   * - ``device``
     - ``"cuda"``
     - Training device, passed to Lightning as the accelerator
       (``"cuda"`` becomes ``"gpu"``).
   * - ``devices``
     - 1
     - Number of devices to train on; ``-1`` uses every visible device.
       More than one trains with DDP. See `Training on several devices`_.
   * - ``verbose``
     - ``false``
     - Print the run's setup and a table with one row per epoch (the
       columns ``Epoch``, ``Iteration``,
       ``Training Time``, ``Validation Time``, ``Training MNLL``,
       ``Training Count MSE``, ``Validation MNLL``, ``Validation Profile
       Pearson``, ``Validation Count Pearson``, ``Validation Count MSE``,
       ``Validation Count Pearson (Peaks+Negatives)``, ``Validation Count
       MSE (Peaks+Negatives)``, ``Validation AUROC``, ``Validation AUPRC``
       and ``Saved?``), and allow a progress bar.
   * - ``progress_bar``
     - ``null``
     - With ``verbose``, whether to draw Lightning's progress bar, with the
       latest validation profile and count Pearson to its right. ``null``
       draws it only when stdout is a terminal or a Jupyter kernel, so a
       run redirected to a file logs just the table. Several runs sharing
       one terminal overwrite each other's bars; set ``false`` for them.
   * - ``skip``
     - ``false``
     - No-op the command.

Training runs through :func:`cherimoya.training.fit` and writes four
files next to ``name``:

* ``<name>.torch``: the EMA weights from the epoch with the highest
  mean validation count Pearson.
* ``<name>.final.torch``: the EMA weights at the end of training.
* ``<name>.log``: one tab-separated row per epoch of training and
  validation measures, with the columns listed under ``verbose`` above.
* ``<name>.detailed.log``: the same, plus one profile Pearson, count
  Pearson, AUROC and AUPRC column per signal group.

Validation uses the ``loci`` and every ``negatives`` locus on the
``validation_chroms``. The validation profile measures, the count
Pearson and MSE, and the checkpoint and early-stopping criterion are
computed on the peaks alone. The four columns before ``Saved?`` are the
count Pearson and MSE over peaks and negatives together, and the AUROC and
AUPRC of the predicted log counts at separating peaks from negatives,
each averaged over the signal groups. They are empty when
``negatives`` is ``null`` or has no locus on the validation
chromosomes. The AUPRC depends on the ratio of peaks to negatives in
the validation set.

Both checkpoints load with :meth:`cherimoya.Cherimoya.load`.
:doc:`multi_task` describes the two logs and how the per-group averages
are formed.


.. _cli-several-devices:

Training on several devices
~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``devices`` other than 1, training uses DDP. ``batch_size`` is
the global batch and must be divisible by the number of devices; each
device takes an equal contiguous slice of every global batch, so
each step sees exactly the examples one device would. A trailing
partial global batch is dropped, on one device or several. Validation
is split across the devices without padding, and the metrics are
computed over the whole validation set.

Lightning starts every rank after the first by re-running the current
command, so everything in ``cherimoya fit`` before training runs once
per rank; only rank 0 prints and runs the evaluate step. In a sweep,
each job's ranks rerun that job alone. With ``random_state`` set to
``null``, the ranks Lightning launches use the seed rank 0 drew. In ``cherimoya pipeline``, when ``fit.devices`` is
not 1 the fit step runs as a separate
``python -m cherimoya_cli fit -p <name>.fit.yaml`` process, so that the
re-run command is the fit rather than the whole pipeline.

What changes and what does not:

* **Which GPUs.** ``devices=N`` trains on the first ``N`` GPUs that
  are visible to the process; choose them with
  ``CUDA_VISIBLE_DEVICES``, e.g.
  ``CUDA_VISIBLE_DEVICES=2,3 cherimoya fit -p run.yaml devices=2``.
* **Memory.** Each GPU holds ``batch_size / devices`` training
  examples, so the global batch can grow with the number of GPUs.
  Every rank loads the whole training and validation set into host
  memory and runs ``num_workers`` loader workers of its own, so host
  memory and worker count scale with ``devices``.
* **The schedule.** A step is one global batch whatever the number of
  devices, so ``max_epochs``, ``min_total_steps`` and the warmup and
  decay lengths mean the same thing on one GPU or several.
* **The outputs.** The checkpoints, ``<name>.log``,
  ``<name>.detailed.log`` and the per-epoch table under ``verbose`` are
  written by rank 0 and have the same format as on one device.
* **The numbers.** Each step sees the same examples as on one device,
  but each GPU runs a smaller batch, which changes which kernels run
  at TF32 and bf16 precision; see :ref:`what a seed fixes
  <reproducibility>`.

Multi-GPU training needs a terminal or a batch script. Lightning
refuses the ``ddp`` strategy inside a Jupyter notebook, so
:func:`cherimoya.training.fit` with more than one device raises there.

**Under SLURM**, Lightning takes the processes from SLURM instead of
starting them itself. Request one task per GPU on a single node and
launch ``cherimoya fit`` with ``srun``::

   #SBATCH --nodes=1
   #SBATCH --gres=gpu:4
   #SBATCH --ntasks-per-node=4

   srun cherimoya fit -p run.yaml devices=4

Lightning raises an error when ``--ntasks-per-node`` differs from
``devices``. ``srun`` starts every rank at once, so with
``random_state`` set to ``null`` there is no draw for the others to
inherit; each rank derives the same seed from ``SLURM_JOB_ID`` and
``SLURM_STEP_ID`` instead, and rank 0 prints it. Run ``cherimoya fit``
rather than ``cherimoya pipeline`` under ``srun``, since every task
runs the whole command.


.. _cli-evaluate:

cherimoya evaluate
------------------

Score a trained model on held-out chromosomes and write one TSV row per
signal group.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Key
     - Default
     - Description
   * - ``model``
     - required
     - Path to a saved ``.torch`` checkpoint.
   * - ``sequences``
     - required
     - Reference genome FASTA.
   * - ``loci``
     - required
     - BED file(s) of evaluation loci.
   * - ``negatives``
     - ``null``
     - BED file(s) of negatives, scored with the loci for the columns
       that use negatives.
   * - ``signals``
     - required
     - Signal bigWigs to score against (must match training). Accepts
       the same flat-or-grouped form as ``fit``'s ``signals``. The
       per-group count pooling used to compute count metrics is
       recovered from the loaded model's checkpoint, so passing the
       grouped form is recommended but not required.
   * - ``controls``
     - ``null``
     - Control bigWigs (must match training). Same grouping rule as
       ``signals``.
   * - ``exclusion_lists``
     - ``null``
     - BED file(s) of regions to exclude.
   * - ``chroms``
     - ``[chr8, chr20]``
     - Held-out chromosomes.
   * - ``batch_size``
     - 512
     - Inference batch size.
   * - ``in_window`` / ``out_window``
     - 2114 / 1000
     - Window sizes (must match training).
   * - ``reverse_complement_average``
     - ``false``
     - Run predictions on RC inputs and average the results. The RC
       swaps the strands within each group, as training does: the
       signal grouping comes from the checkpoint and the control
       grouping from ``controls``.
   * - ``summits``
     - ``false``
     - Center the loci on the narrowPeak summit column, as ``fit`` does
       with the same key. Negatives are always midpoint-centered.
   * - ``compile`` / ``compile_mode``
     - ``true`` / ``"max-autotune"``
     - Passed through to :meth:`cherimoya.Cherimoya.load`. ``compile``
       wraps the forward in ``torch.compile``; set it to ``false`` for
       an eager forward, the fix for a ``torch.compile`` or CUDA-graph
       error. ``compile_mode`` is the ``mode`` passed to
       ``torch.compile``; useful alternatives are
       ``"max-autotune-no-cudagraphs"`` (same kernel autotuning, no
       CUDA-graph capture) and ``"reduce-overhead"``.
   * - ``dtype`` / ``device``
     - ``"float32"`` / ``"cuda"``
     - Inference dtype and device.
   * - ``verbose``
     - ``false``
     - Print progress.
   * - ``performance_filename``
     - ``"performance.tsv"``
     - Output TSV.
   * - ``skip``
     - ``false``
     - No-op the command.

The TSV columns are
``profile_mnll``, ``profile_jsd``, ``profile_pearson``,
``profile_spearman``, ``count_pearson``, ``count_spearman``,
``count_mse``, computed on the ``loci`` alone, then
``all_count_pearson``, ``all_count_spearman``, ``all_count_mse``,
computed on the loci and negatives together, and ``auroc`` and
``auprc``, for the predicted log counts separating the loci from the
negatives. The last five are ``nan`` without negatives. The evaluate
configs ``fit`` writes carry its ``negatives``, so the evaluations
after training include them. When no locus falls on ``chroms``,
``evaluate`` prints a message and writes no file. The file has one data
row per signal group, in ``signal_groups`` order. For a single-group model (the default) this
is a single row holding the same per-group mean that
``calculate_performance_measures`` returns; for a multi-group model
row ``i`` corresponds to ``signal_groups[i]``. Profile metrics are
the mean of the metric over (validation loci × the group's
channels); count metrics are read directly from the per-group
``(n_groups,)`` tensors. See :doc:`multi_task` for an in-depth
description.


cherimoya attribute
-------------------

Compute DeepLIFT/SHAP attributions (the default) or saturation
mutagenesis with ``algorithm=saturation_mutagenesis``; see
:doc:`tutorials/attribution`.

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Key
     - Default
     - Description
   * - ``model``
     - required
     - Path to a saved ``.torch`` checkpoint.
   * - ``sequences``
     - required
     - Reference genome FASTA.
   * - ``loci``
     - required
     - BED file(s) of loci to attribute.
   * - ``exclusion_lists``
     - ``null``
     - BED file(s) of regions to exclude. Loci whose extraction window
       overlaps one are not attributed, and ``idx_filename`` records
       which were dropped.
   * - ``chroms``
     - training + validation chroms
     - Chromosomes to attribute.
   * - ``algorithm``
     - ``"deep_lift_shap"``
     - ``"deep_lift_shap"`` (DeepLIFT/SHAP against dinucleotide-shuffled
       references, with Cherimoya's DeepLIFT rules registered) or
       ``"saturation_mutagenesis"``. Both write arrays of the same
       shape.
   * - ``output``
     - ``"counts"``
     - Attribute to counts or profile (``"profile"``).
   * - ``group``
     - 0
     - Index into the model's ``signal_groups`` of the one group to
       attribute. ``null`` attributes every group at once; DeepLIFT/SHAP
       rejects that for ``"counts"`` on a model with more than one group,
       since it attributes a single output.
   * - ``attr_window``
     - 400
     - Width of the centred slice that is actually attributed, and the
       width of the arrays written to ``ohe_filename`` and
       ``attr_filename``. Saturation mutagenesis is one forward pass
       per alternate base per position, so for it this sets the cost
       of the step; DeepLIFT/SHAP attributes the whole ``in_window``
       and keeps this slice. Must not exceed ``in_window``.
   * - ``n_shuffles``
     - 20
     - DeepLIFT/SHAP: dinucleotide-shuffled references per sequence.
   * - ``warning_threshold``
     - 0.001
     - DeepLIFT/SHAP: warn when an example's attributions miss the
       change in prediction by more than this.
   * - ``random_state``
     - 0
     - DeepLIFT/SHAP: seed for the shuffled references.
   * - ``batch_size``
     - 64
     - Batch size. For DeepLIFT/SHAP this counts sequence-reference
       pairs, each run forward and backward.
   * - ``in_window``
     - 2114
     - Width of the sequence window extracted per locus. Must match the
       window the model was trained at.
   * - ``compile`` / ``compile_mode``
     - ``false`` / ``"max-autotune"``
     - Whether to ``torch.compile`` the model, for saturation mutagenesis
       only; DeepLIFT/SHAP always loads it uncompiled, since its backward
       hooks cause graph breaks and recompiles. Off by default because neither
       algorithm ran faster compiled, and compiling added 6–70 s to the
       first call.
   * - ``dtype`` / ``device``
     - ``"float32"`` / ``"cuda"``
     - Inference dtype and device.
   * - ``verbose``
     - ``false``
     - Print progress.
   * - ``ohe_filename``
     - ``"attributions.ohe.npz"``
     - Output: one-hot encoded inputs, ``attr_window`` wide.
   * - ``attr_filename``
     - ``"attributions.attr.npz"``
     - Output: per-base hypothetical importance.
   * - ``idx_filename``
     - ``"attributions.idx.npy"``
     - Output: boolean mask back to the original loci list.
   * - ``skip``
     - ``false``
     - No-op the command.


cherimoya seqlets
-----------------

Call seqlets from the arrays ``attribute`` wrote. ``loci`` and
``chroms`` must match the attribute run, since they convert
example-relative seqlet coordinates back to genome coordinates. Loci
that ``attribute`` excluded are already marked in its index file, so
``seqlets`` takes no ``exclusion_lists``.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Key
     - Default
     - Description
   * - ``loci``
     - required
     - BED file(s) the attributions were computed on.
   * - ``ohe_filename`` / ``attr_filename`` / ``idx_filename``
     - required
     - The one-hot, attribution and index files from ``attribute``.
   * - ``chroms``
     - training + validation chroms
     - Chromosomes the attributions were computed on.
   * - ``threshold``
     - 0.01
     - Recursive seqlet p-value threshold.
   * - ``min_seqlet_len`` / ``max_seqlet_len``
     - 4 / 25
     - Minimum and maximum seqlet length (bp).
   * - ``additional_flanks``
     - 3
     - Flanking bases retained on each side.
   * - ``verbose``
     - ``false``
     - Not read.
   * - ``output_filename``
     - ``"seqlets.bed"``
     - Output BED.
   * - ``skip``
     - ``false``
     - No-op the command.

The emitted BED is in the same coordinate system as ``loci``. Seqlet
positions are relative to the attribution array, which covers a slice
centred on each locus rather than the full extraction window, so every
seqlet falls inside the slice that ``attribute`` scored.


cherimoya marginalize
---------------------

Insert each motif into background sequences and report the predicted
effect.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Key
     - Default
     - Description
   * - ``model``
     - required
     - Path to a saved ``.torch`` checkpoint.
   * - ``sequences``
     - required
     - Reference genome FASTA.
   * - ``motifs``
     - required
     - MEME-format motif file.
   * - ``loci``
     - required
     - BED file(s) of background loci to insert motifs into.
   * - ``exclusion_lists``
     - ``null``
     - BED file(s) of regions to exclude. Background loci whose window
       overlaps one are not used.
   * - ``chroms``
     - training chroms
     - Chromosomes to draw background loci from.
   * - ``n_loci``
     - 100
     - Number of background loci per motif. ``null`` uses every locus.
   * - ``shuffle``
     - ``false``
     - Draw the ``n_loci`` background loci at random from the whole
       file rather than taking the first ``n_loci`` rows. Because a
       sample cannot be drawn without seeing the population, this reads
       every locus in ``loci`` into memory before selecting; the
       unshuffled path stops at ``n_loci`` and does not.
   * - ``attributions``
     - ``false``
     - Compute attributions on the inserted motif.
   * - ``minimal``
     - ``true``
     - Use the minimal marginalization output format.
   * - ``random_state``
     - 0
     - RNG seed for the locus shuffle.
   * - ``batch_size``
     - 512
     - Inference batch size.
   * - ``in_window``
     - 2114
     - Width of the background sequence extracted per locus. Must match
       the window the model was trained at.
   * - ``out_window``
     - 1000
     - Not read.
   * - ``compile`` / ``compile_mode``
     - ``true`` / ``"max-autotune"``
     - As for ``evaluate``.
   * - ``device``
     - ``"cuda"``
     - Inference device.
   * - ``verbose``
     - ``false``
     - Print progress.
   * - ``output_filename``
     - ``"marginalize/"``
     - Output directory.
   * - ``skip``
     - ``false``
     - No-op the command.


cherimoya negatives
-------------------

Sample GC-matched negative regions for a peak file.

.. code-block:: bash

   cherimoya negatives peaks=peaks.narrowPeak fasta=hg38.fa \
       output=negatives.bed

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Key
     - Default
     - Description
   * - ``peaks``
     - required
     - Peak BED.
   * - ``fasta``
     - required
     - Reference genome FASTA.
   * - ``output``
     - required
     - Output BED.
   * - ``bigwig``
     - ``null``
     - Signal bigWig, used with ``beta`` to drop negatives with too
       much signal.
   * - ``bin_width``
     - 0.02
     - GC bin width to match.
   * - ``max_n_perc``
     - 0.1
     - Maximum fraction of ``N`` bases allowed per locus.
   * - ``beta``
     - 0.5
     - Multiplier on the minimum peak counts when filtering negatives
       by signal.
   * - ``in_window``
     - 2114
     - Window over which GC content is calculated.
   * - ``out_window``
     - 1000
     - Non-overlapping stride.
   * - ``verbose``
     - ``false``
     - Print progress.


cherimoya pipeline
------------------

Run every step on the given files: peak calling, bigWig conversion,
negative sampling, training and evaluation, attribution, seqlet
calling, seqlet annotation, TF-MoDISco and marginalization. See
:doc:`tutorials/cli_pipeline` for what each step does and writes.

The top-level keys are shared. Each step has its own section (``fit``,
``attribute``, ...) typed by that command's schema, and many of its
defaults point at a top-level key, so ``name=x`` or ``in_window=3000``
reaches every step while ``fit.batch_size=32`` changes only one.
Before each step runs, the pipeline saves its section, with the links
filled in, as ``<name>.<command>.yaml``.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Key
     - Default
     - Description
   * - ``name``
     - required
     - Prefix of every output file.
   * - ``sequences``
     - required
     - Reference genome FASTA.
   * - ``loci``
     - required
     - List of peak BED files, or a single path. ``null`` calls peaks
       with MACS3.
   * - ``negatives``
     - required
     - List of negative BED files, or a single path. ``null`` samples
       GC-matched negatives from the first ``loci`` file. Those on the
       validation chromosomes are also scored in validation and
       evaluation (see `cherimoya fit`_), and marginalization inserts
       motifs into them.
   * - ``signals``
     - required
     - Signal files: BAM, SAM, fragment BED/TSV, or bigWig. Flat or
       grouped (see above). Non-bigWig files are converted with
       ``bam2bw`` first.
   * - ``controls``
     - ``null``
     - Control files. Same grouping and conversion rules as
       ``signals``.
   * - ``exclusion_lists``
     - ``null``
     - BED file(s) of regions to exclude.
   * - ``motifs``
     - ``null``
     - MEME-format motif database. When set, seqlets are annotated with
       tomtom-lite, the MoDISco report is matched against it, and
       marginalization runs. ``null`` skips annotation and
       marginalization.
   * - ``model``
     - ``null``
     - Existing ``.torch`` checkpoint. When set, training is skipped
       and later steps use this model.
   * - ``in_window`` / ``out_window``
     - 2114 / 1000
     - Window sizes (bp).
   * - ``batch_size``
     - 512
     - Batch size for marginalization. The fit (64) and attribute (64)
       steps keep their own.
   * - ``random_state``
     - 0
     - Seed shared by fit, attribute and marginalize. ``null`` draws a
       seed in fit, prints it and saves it in ``<name>.fit.yaml``;
       attribute and marginalize then get ``null``.
   * - ``compile`` / ``compile_mode``
     - ``true`` / ``"max-autotune"``
     - ``compile`` reaches fit and marginalize; the attribute step keeps
       its own (``false``). ``compile_mode`` reaches fit, attribute and
       marginalize. In fit they apply to the training model and to both
       evaluations.
   * - ``dtype`` / ``device``
     - ``"float32"`` / ``"cuda"``
     - Reach fit and attribute; ``device`` also reaches marginalize.
   * - ``verbose``
     - ``true``
     - Print per-step progress.
   * - ``skip``
     - ``false``
     - No-op the whole pipeline. To skip one step, set that step's
       ``skip``, such as ``attribute.skip=true``.
   * - ``dry_run``
     - ``false``
     - Write the per-step YAML files and run nothing.

Step sections
~~~~~~~~~~~~~

The step sections take every key of the matching command (see the
tables above), with these defaults changed:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Section
     - Defaults that differ from the command
   * - ``negative_sampling``
     - ``peaks: ${loci.0}``, ``fasta: ${sequences}``,
       ``output: ${name}.negatives.bed``, ``in_window``, ``out_window``,
       ``verbose: ${preprocessing.verbose}``. Runs only when
       ``negatives`` is ``null``.
   * - ``fit``
     - ``name``, ``sequences``, ``loci``, ``negatives``, ``signals``,
       ``controls``, ``exclusion_lists``, ``in_window``, ``out_window``,
       ``random_state``, ``compile``, ``compile_mode``, ``dtype``,
       ``device`` and ``verbose`` from the top level. Runs only when
       ``model`` is ``null``, and writes ``<name>.torch``. With
       ``fit.devices`` other than 1 it runs as its own process (see
       `Training on several devices`_).
   * - ``attribute``
     - ``model``, ``sequences``, ``loci``, ``exclusion_lists``,
       ``in_window``, ``random_state``, ``compile_mode``, ``dtype``,
       ``device`` and ``verbose`` from the top level;
       ``ohe_filename: ${name}.attributions.ohe.npz``,
       ``attr_filename: ${name}.attributions.attr.npz``,
       ``idx_filename: ${name}.attributions.idxs.npy``.
   * - ``seqlets``
     - ``loci`` and ``verbose`` from the top level; ``chroms`` and the
       three input files from ``attribute``;
       ``output_filename: ${name}.seqlets.bed``.
   * - ``marginalize``
     - ``model``, ``sequences``, ``motifs``, ``exclusion_lists``,
       ``in_window``, ``out_window``, ``batch_size``, ``random_state``,
       ``compile``, ``compile_mode``, ``device`` and ``verbose`` from
       the top level; ``loci: ${negatives}``, the background loci;
       ``output_filename: ${name}_marginalize/``. Runs only when
       ``motifs`` is set.

``preprocessing``
~~~~~~~~~~~~~~~~~

MACS3 peak calling and ``bam2bw`` conversion.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Key
     - Default
     - Description
   * - ``unstranded``
     - ``false``
     - Produce a single unstranded bigWig instead of a ``+ / -`` pair.
   * - ``fragments``
     - ``false``
     - Treat input as fragment files.
   * - ``paired_end``
     - ``false``
     - Treat input as paired-end; affects the MACS3 format
       (``BAMPE``).
   * - ``pos_shift``
     - 0
     - + strand shift (bp).
   * - ``neg_shift``
     - 0
     - - strand shift (bp).
   * - ``scale_factor``
     - 1.0
     - Multiplier on raw counts of the signal files.
   * - ``read_depth``
     - ``false``
     - Pass ``-r`` to ``bam2bw`` to scale by sequencing depth.
   * - ``callpeaks_format``
     - ``null``
     - MACS3 ``-f`` value. ``null`` detects it from the first signal
       file's extension and ``paired_end``, or uses ``FRAG`` when
       ``fragments`` is set.
   * - ``callpeaks_gsize``
     - ``"hs"``
     - MACS3 ``-g`` value (effective genome size). Use ``"mm"`` for
       mouse, a numeric value for other organisms.
   * - ``callpeaks_q``
     - 0.05
     - MACS3 q-value cutoff.
   * - ``verbose``
     - ``true``
     - Print preprocessing progress. Also the default ``verbose`` of
       ``negative_sampling``.

``annotation``
~~~~~~~~~~~~~~

tomtom-lite (``ttl``) annotation of the seqlets. Runs only when
``motifs`` is set.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Key
     - Default
     - Description
   * - ``sequences``
     - ``${sequences}``
     - Reference genome FASTA.
   * - ``seqlet_filename``
     - ``${seqlets.output_filename}``
     - Seqlet BED from the seqlets step.
   * - ``motifs``
     - ``${motifs}``
     - MEME-format motif database.
   * - ``n_score_bins``
     - 100
     - ``ttl -s``.
   * - ``n_median_bins``
     - 1000
     - ``ttl -m``.
   * - ``n_target_bins``
     - 100
     - ``ttl -a``.
   * - ``n_cache``
     - 250
     - ``ttl -c``.
   * - ``reverse_complement``
     - ``true``
     - Scan motifs in both orientations.
   * - ``n_jobs``
     - -1
     - Parallel workers; -1 uses all cores.
   * - ``output_filename``
     - ``${name}.seqlets_annotated.bed``
     - Output BED.
   * - ``skip``
     - ``false``
     - Skip the annotation.

``modisco_motifs`` / ``modisco_report``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Key
     - Default
     - Description
   * - ``modisco_motifs.n_seqlets``
     - 100000
     - Number of seqlets passed to ``modisco motifs``.
   * - ``modisco_motifs.output_filename``
     - ``${name}_modisco_results.h5``
     - HDF5 output of ``modisco motifs``.
   * - ``modisco_motifs.verbose``
     - ``${verbose}``
     - Pass ``-v`` to ``modisco motifs``.
   * - ``modisco_report.output_folder``
     - ``${name}_modisco/``
     - Directory output of ``modisco report``.
   * - ``modisco_report.motifs``
     - ``${motifs}``
     - Motif database passed to ``modisco report -m``.
   * - ``modisco_report.verbose``
     - ``${verbose}``
     - Print the step header.


cherimoya install-skill
-----------------------

Install the bundled Cherimoya agent skill for `Claude Code
<https://claude.com/claude-code>`_ into your skills directory, creating
``cherimoya/`` inside it. The skill teaches the assistant to drive this CLI
and the Python API: working out which inputs you have, choosing
assay-appropriate settings, calling the right subcommands, and interpreting
outputs. It also teaches it to ask clarifying questions when an input is ambiguous.

This command takes flags, not config keys:

* ``-d, --directory``: skills directory to install into. Default
  ``~/.claude/skills``.
* ``--symlink``: symlink the packaged skill instead of copying it, so
  in-place edits are reflected without reinstalling. Breaks if the install
  location moves.
* ``-f, --force``: overwrite an existing installation at the destination.

Restart Claude Code (or reload skills) to pick it up.
