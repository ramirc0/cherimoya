Quickstart
==========

This page shows the two main ways to use Cherimoya: via the **command-line
pipeline** or via the **Python API**.


Using the CLI Pipeline
----------------------

The fastest way to go from raw data to trained model and motif analysis is the
end-to-end pipeline. You need:

1. A genome FASTA file (e.g., ``hg38.fa``)
2. One or more signal files (BAM, SAM, BED, or bigWig)
3. (Optional) Control signal files

**Step 1: Generate a pipeline JSON**

.. code-block:: bash

   cherimoya pipeline-json \
       -s hg38.fa \
       -i signal.bam \
       -n my_experiment \
       -o my_experiment.pipeline.json

**Step 2: Run the full pipeline**

.. code-block:: bash

   cherimoya pipeline -p my_experiment.pipeline.json

This will automatically:

- Call peaks using MACS3
- Convert BAM files to bigWig format
- Sample GC-matched negative regions
- Train a Cherimoya model
- Calculate attributions
- Identify seqlets
- Run TF-MoDISco motif discovery


Using the Python API
--------------------

For more control, use the Python API directly.

**Instantiate a model:**

.. code-block:: python

   from cherimoya import Cherimoya

   model = Cherimoya(
       n_filters=96,       # Number of convolutional filters (default 96)
       n_layers=9,         # Number of Cheri Blocks
       n_outputs=2,        # Number of output tracks (e.g., 2 for stranded)
       n_control_tracks=0, # Number of control tracks (0 if no controls)
   ).cuda()

**Load training data:**

.. code-block:: python

   from cherimoya.io import PeakGenerator

   training_data = PeakGenerator(
       peaks="peaks.narrowPeak",
       negatives="negatives.bed",
       sequences="hg38.fa",
       signals=["signal.+.bw", "signal.-.bw"],
       chroms=["chr1", "chr2", "chr3"],  # Training chromosomes
       in_window=2114,
       out_window=1000,
       max_jitter=128,
       batch_size=64,
   )

**Set up optimizers and train:**

.. code-block:: python

   from torch.optim import AdamW, Muon
   from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

   # Separate parameters for Muon (2D weights) and AdamW (everything else)
   muon_params, adam_params = [], []
   for name, p in model.named_parameters():
       if p.ndim == 2 and "weight" in name and name != "linear.weight":
           muon_params.append(p)
       else:
           adam_params.append(p)

   muon_optimizer = Muon(muon_params, lr=0.01)
   adam_optimizer = AdamW(adam_params, lr=0.004)

   # Warmup + cosine decay schedules
   n_warmup = len(training_data) * 5
   n_total = len(training_data) * 50

   muon_scheduler = SequentialLR(muon_optimizer, schedulers=[
       LinearLR(muon_optimizer, start_factor=0.01, total_iters=n_warmup),
       CosineAnnealingLR(muon_optimizer, T_max=n_total, eta_min=1e-5),
   ], milestones=[n_warmup])

   adam_scheduler = SequentialLR(adam_optimizer, schedulers=[
       LinearLR(adam_optimizer, start_factor=0.01, total_iters=n_warmup),
       CosineAnnealingLR(adam_optimizer, T_max=n_total, eta_min=1e-5),
   ], milestones=[n_warmup])

   # Train
   model.fit(
       training_data,
       muon_optimizer, adam_optimizer,
       muon_scheduler, adam_scheduler,
       X_valid=X_valid,
       X_ctl_valid=None,
       y_valid=y_valid,
       max_epochs=50,
       batch_size=64,
   )

**Make predictions:**

.. code-block:: python

   from tangermeme.predict import predict

   y_profile, y_counts = predict(
       model, X_test,
       batch_size=64,
       device='cuda',
   )


Next Steps
----------

- :doc:`architecture` — understand the Cheri Block and model design
- :doc:`tutorials/cli_pipeline` — detailed CLI pipeline walkthrough
- :doc:`tutorials/python_api` — full Python API tutorial
- :doc:`tutorials/attribution` — attribution and motif analysis
