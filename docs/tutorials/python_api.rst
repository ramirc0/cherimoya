Python API Tutorial
===================

This tutorial demonstrates how to use the Cherimoya Python API to train a
model, make predictions, and evaluate performance.


Creating a Model
----------------

A Cherimoya model is a standard ``torch.nn.Module``:

.. code-block:: python

   from cherimoya import Cherimoya

   model = Cherimoya(
       n_filters=96,           # Width of the convolutional backbone (default 96)
       n_layers=9,             # Number of Cheri Blocks (dilation: 1, 2, ..., 256)
       n_outputs=2,            # Output tracks (2 for stranded data)
       n_control_tracks=0,     # Control input tracks (0 if no controls)
       expansion=2,            # MLP expansion factor inside each Cheri Block (default 2)
       residual_scale=0.15,    # Fixed scalar on each residual connection (default 0.15)
       single_count_output=True,  # Single scalar count vs. per-track counts
       name="my_model",        # Model name (used for save filenames)
   ).cuda()


Understanding the Input/Output Shapes
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Tensor
     - Shape
     - Description
   * - ``X`` (input)
     - ``(batch, 4, in_window)``
     - One-hot encoded DNA sequence
   * - ``X_ctl`` (optional)
     - ``(batch, n_control, in_window)``
     - Control signal per position
   * - ``y_profile`` (output)
     - ``(batch, n_outputs, out_window)``
     - Predicted profile logits
   * - ``y_counts`` (output)
     - ``(batch, n_count_outputs)``
     - Predicted log counts

The default ``in_window`` is 2114 bp and ``out_window`` is 1000 bp. The
difference (``trimming = (2114 - 1000) / 2 = 557``) accounts for the receptive
field of the dilated convolution stack.


Loading Training Data
---------------------

The :func:`~cherimoya.io.PeakGenerator` function handles all data I/O:

.. code-block:: python

   from cherimoya.io import PeakGenerator

   training_data = PeakGenerator(
       peaks="peaks.narrowPeak",         # Peak regions (BED/narrowPeak)
       negatives="negatives.bed",        # GC-matched negative regions
       sequences="hg38.fa",             # Reference genome
       signals=["signal.+.bw", "signal.-.bw"],  # Signal tracks
       controls=None,                    # Control tracks (or list of bigWigs)
       chroms=["chr2", "chr4", "chr5"],  # Training chromosomes
       in_window=2114,
       out_window=1000,
       max_jitter=128,                   # Random position jitter (bp)
       negative_ratio=0.1,              # Ratio of negatives to peaks
       reverse_complement=True,          # Augment with reverse complements
       batch_size=64,
       num_workers=1,                    # Async prefetch workers
       random_state=0,                   # Sampler seed (reproducible)
   )

.. tip::

   Use ``verbose=True`` to see progress bars during data loading and summary
   statistics about the number of peaks, negatives, and filtered regions.


Sampling Strategy
^^^^^^^^^^^^^^^^^

The underlying :class:`~cherimoya.io.PeakNegativeSampler` is fully
deterministic given ``random_state``. ``__getitem__(idx)`` is a pure
function of ``idx`` and the current epoch, with no dependence on call
history. As a consequence:

- Each epoch yields exactly ``n_peaks + int(n_peaks * negative_ratio)``
  examples; every peak appears exactly once and the peak/negative
  interleaving is random but reproducible.
- Setting ``num_workers > 1`` produces the **same** sequence of batches
  as ``num_workers = 1`` — just faster. All workers compute the same
  data for any given index.
- Per-position augmentations (jitter, reverse-complement) are also
  drawn from the per-epoch RNG, so two runs with the same seed produce
  bit-identical training data.


Preparing Validation Data
--------------------------

Validation data is loaded as a single block of tensors:

.. code-block:: python

   from tangermeme.io import extract_loci

   valid_data = extract_loci(
       sequences="hg38.fa",
       signals=["signal.+.bw", "signal.-.bw"],
       loci="peaks.narrowPeak",
       chroms=["chr8", "chr20"],  # Held-out validation chromosomes
       in_window=2114,
       out_window=1000,
       max_jitter=0,              # No jitter for validation
       ignore=list('QWERYUIOPSDFHJKLZXVBNM'),  # Ignore non-standard chroms
   )

   X_valid, y_valid = valid_data  # Without controls
   # X_valid, y_valid, X_ctl_valid = valid_data  # With controls


Setting Up Optimizers
---------------------

Cherimoya uses a dual-optimizer strategy — **Muon** for 2D projection layers
and **AdamW** for everything else:

.. code-block:: python

   from torch.optim import AdamW, Muon
   from torch.optim.lr_scheduler import (
       LinearLR, CosineAnnealingLR, SequentialLR
   )

   # Route parameters to the appropriate optimizer, with Muon only handling
   # internal 2D matrices and leaving all other parameters, including the
   # head and tail layers, to AdamW
   muon_params, adam_params = [], []
   for name, p in model.named_parameters():
       if p.ndim == 2 and "weight" in name and name != "linear.weight":
           muon_params.append(p)
       else:
           adam_params.append(p)

   # Create optimizers
   muon_optimizer = Muon(muon_params, lr=0.01, weight_decay=0.0)
   adam_optimizer = AdamW(adam_params, lr=0.004, weight_decay=0.0)

   # Warmup + cosine decay schedules
   n_warmup_iters = len(training_data) * 5      # 5 epoch warmup
   n_total_iters = len(training_data) * 50       # 50 total epochs

   muon_scheduler = SequentialLR(muon_optimizer, schedulers=[
       LinearLR(muon_optimizer, start_factor=0.01, total_iters=n_warmup_iters),
       CosineAnnealingLR(muon_optimizer, T_max=n_total_iters, eta_min=1e-5),
   ], milestones=[n_warmup_iters])

   adam_scheduler = SequentialLR(adam_optimizer, schedulers=[
       LinearLR(adam_optimizer, start_factor=0.01, total_iters=n_warmup_iters),
       CosineAnnealingLR(adam_optimizer, T_max=n_total_iters, eta_min=1e-5),
   ], milestones=[n_warmup_iters])


Training the Model
------------------

.. code-block:: python

   model.fit(
       training_data,
       muon_optimizer, adam_optimizer,
       muon_scheduler, adam_scheduler,
       X_valid=X_valid,
       X_ctl_valid=None,    # Pass control tensors here if using controls
       y_valid=y_valid,
       max_epochs=50,
       batch_size=64,
       early_stopping=15,   # Stop after 15 epochs without improvement
       dtype='float32',     # Or 'bfloat16' for mixed precision
       device='cuda',
   )

During training, the model saves:

- ``my_model.torch`` — best model checkpoint
- ``my_model.final.torch`` — final model after training
- ``my_model.log`` — training/validation metrics log


Saving and Loading Models
-------------------------

Cherimoya checkpoints store the constructor arguments next to the parameter
state dict. This makes loading robust to package layout changes and lets the
checkpoint be loaded with PyTorch's ``weights_only=True`` setting.

**Saving**:

.. code-block:: python

   from cherimoya import Cherimoya

   model = Cherimoya(n_filters=96, n_layers=9, n_outputs=2)
   # ... train ...
   model.save("my_model.torch")

**Loading**:

.. code-block:: python

   from cherimoya import Cherimoya

   # Load to CPU (default) — fine for inspection or CPU-only inference.
   model = Cherimoya.load("my_model.torch")

   # Load directly onto GPU.
   model = Cherimoya.load("my_model.torch", device="cuda")

The checkpoint is a dictionary with two keys, ``config`` (the kwargs
passed to ``Cherimoya.__init__``) and ``state_dict`` (the parameter
tensors). Older checkpoints saved with ``torch.save(model, ...)`` are not
compatible with this loader and should be retrained or migrated.


Making Predictions
------------------

.. code-block:: python

   from tangermeme.predict import predict

   model.eval()
   y_profile, y_counts = predict(
       model, X_test,
       batch_size=64,
       device='cuda',
       dtype='float32',
   )

For reverse-complement averaging (often improves performance):

.. code-block:: python

   import torch

   y_profile_rc, y_counts_rc = predict(
       model, torch.flip(X_test, dims=(-1, -2)),
       batch_size=64, device='cuda',
   )

   y_profile_avg = (y_profile + torch.flip(y_profile_rc, dims=(-1, -2))) / 2
   y_counts_avg = (y_counts + y_counts_rc) / 2


Evaluating Performance
----------------------

.. code-block:: python

   from cherimoya.performance import calculate_performance_measures

   measures = calculate_performance_measures(
       y_profile, y_valid, y_counts,
       measures=['profile_pearson', 'count_pearson', 'profile_jsd'],
   )

   for name, values in measures.items():
       print(f"{name}: {values.mean():.4f}")
