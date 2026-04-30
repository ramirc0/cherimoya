Attribution & Motif Analysis
============================

This tutorial covers how to calculate sequence attributions, identify important
subsequences (seqlets), and discover motifs using TF-MoDISco.


What Are Attributions?
----------------------

Attributions quantify how much each base pair in the input sequence contributes
to the model's prediction. Cherimoya uses **saturation mutagenesis** — the
effect of every possible single-nucleotide mutation — to compute importance
scores.

These scores can be used to:

- Identify transcription factor binding sites
- Discover de novo motifs
- Understand regulatory grammar


Calculating Attributions via CLI
--------------------------------

.. code-block:: bash

   cherimoya attribute -p attribute_params.json

Example ``attribute_params.json``:

.. code-block:: json

   {
       "model": "my_model.torch",
       "sequences": "/path/to/hg38.fa",
       "loci": "peaks.narrowPeak",
       "chroms": ["chr2", "chr4", "chr5"],
       "output": "counts",
       "batch_size": 512,
       "device": "cuda",
       "ohe_filename": "attributions.ohe.npz",
       "attr_filename": "attributions.attr.npz",
       "idx_filename": "attributions.idx.npy"
   }

The ``output`` parameter controls what the attributions are calculated with
respect to:

- ``"counts"`` — attribute to total predicted counts (recommended for most
  analyses)
- ``"profile"`` — attribute to the predicted profile shape


Calculating Attributions via Python
-----------------------------------

.. code-block:: python

   import torch
   from tangermeme.saturation_mutagenesis import saturation_mutagenesis
   from bpnetlite.bpnet import ControlWrapper, CountWrapper

   from cherimoya import Cherimoya

   # Load model
   model = Cherimoya.load("my_model.torch", device="cuda")

   # Wrap for count-based attribution
   if model.n_control_tracks > 0:
       model = ControlWrapper(model)
   wrapper = CountWrapper(model)

   # Calculate attributions (hypothetical importance scores) over the
   # central 400 bp window of each sequence.
   mid = X_sequences.shape[-1] // 2
   X_attr = saturation_mutagenesis(
       wrapper, X_sequences,
       batch_size=512,
       device='cuda',
       hypothetical=True,
       start=mid - 200,
       end=mid + 200,
   )


Identifying Seqlets
-------------------

Seqlets are contiguous subsequences with high attribution scores that likely
correspond to functional elements like transcription factor binding motifs.

**Via CLI:**

.. code-block:: bash

   cherimoya seqlets -p seqlet_params.json

**Via Python:**

.. code-block:: python

   from tangermeme.seqlet import recursive_seqlets

   # Combine one-hot encoding and attributions
   importance = (X_attr * X_ohe).sum(dim=1)

   seqlets = recursive_seqlets(
       importance,
       threshold=0.01,
       min_seqlet_len=4,
       max_seqlet_len=25,
       additional_flanks=3,
   )


TF-MoDISco Motif Discovery
---------------------------

TF-MoDISco groups similar seqlets into motif patterns. This is run
automatically as part of the pipeline, or manually:

.. code-block:: bash

   modisco motifs \
       -s attributions.ohe.npz \
       -a attributions.attr.npz \
       -n 100000 \
       -o modisco_results.h5

   modisco report \
       -i modisco_results.h5 \
       -o modisco_report/ \
       -s ./


Marginalization Experiments
---------------------------

Marginalization experiments measure the causal effect of motif instances by
inserting them into neutral backgrounds:

.. code-block:: bash

   cherimoya marginalize -p marginalize_params.json

This produces a report showing how each motif affects the predicted profile and
counts when inserted into negative (non-peak) sequences.

.. note::

   Marginalization requires a motif database file in the MEME format. Most
   databases will provide a file of this format. If you are looking for a
   database to use, we recommend JASPAR2026.