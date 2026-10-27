Algorithms & Architecture
=========================

thotbook-ai implements a complete **CPU-only LLM training and inference
pipeline** using cutting-edge numerical algorithms from the `optimiz-rs`
library and a high-performance data lakehouse via `Polarway`.

This section details the core algorithms, their mathematical foundations,
and how they enable GPU-free training of billion-parameter language models.

.. toctree::
   :maxdepth: 2
   :caption: Algorithm Modules

   algorithms/training
   algorithms/inference
   algorithms/embeddings
   algorithms/rag
   algorithms/orchestration
   algorithms/variational_calculus
   algorithms/membership_inference

Design Philosophy
-----------------

The system is built on three pillars:

1. **Memory-efficient training** via LoRA adapters and gradient checkpointing
2. **Cache-optimized inference** using KV-cache quantization
3. **Lakehouse-native data flow** through Apache Arrow and Delta Lake

.. math::

   \text{Memory}(\text{LoRA})
   \;=\;
   \mathcal{O}(r \cdot d)
   \;\ll\;
   \mathcal{O}(d^2)
   \;=\;
   \text{Memory}(\text{Full})

where :math:`r \ll d` is the LoRA rank (typically 8–64) and :math:`d` is the
hidden dimension (4096 for Qwen3-8B).

Integration with optimiz-rs
---------------------------

thotbook-ai leverages several modules from `optimiz-rs` for numerical
optimization:

**Sparse Optimization** (``optimiz_rs::sparse``)
   Used in the training loop for L1-regularized LoRA parameter updates,
   enabling sparse adapters that require less memory.

**Stochastic Control** (``optimiz_rs::control``)
   The learning rate scheduler implements a cosine annealing schedule
   derived from optimal control theory:

   .. math::

      \eta_t
      \;=\;
      \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})
      \left(1 + \cos\left(\frac{t}{T}\pi\right)\right)

**Matrix Operations** (``optimiz_rs::linalg``)
   BLAS-3 level matrix multiplications for attention computation,
   optimized for Apple Silicon and x86-64 AVX-512.

**Topology** (``optimiz_rs::topology``)
   Used in the RAG module for persistent homology-based document
   similarity, providing better retrieval than cosine similarity alone.

**BSDE / Mean-Field Games** (``optimiz_rs::bsde``, ``optimiz_rs::mfg``)
   The learning-rate schedule and the ``llm-auto`` self-training loop are
   both functional-minimisation problems in disguise — see
   :doc:`algorithms/variational_calculus` for the exact correspondence
   between the Crank–Nicolson :math:`\theta`-scheme / Fokker–Planck–HJB
   sweep and the discrete training dynamics.

**Differential Evolution / MMD / Mutual Information**
   (``optimiz_rs::differential_evolution``, ``optimiz_rs::mmd_gaussian``,
   ``optimiz_rs::mutual_information``)
   Used by the membership-inference privacy audit
   (:doc:`algorithms/membership_inference`) to calibrate attack decision
   thresholds and to test member/non-member score distinguishability.

Integration with Polarway
-------------------------

The Polarway lakehouse provides:

1. **Delta Lake tables** for versioned, time-travel-capable data storage
2. **Apache Arrow IPC** for zero-copy tensor serialization
3. **Incremental ingestion** with exactly-once semantics

Data flow:

.. figure:: /_static/diagrams/thotbook_pipeline.*
   :alt: thotbook-ai pipeline: ingestion (M1 corpus ingest, M2 tokenize)
         feeds training (M3 LoRA, M4 RAG/HNSW) feeds serving (M5
         OpenAI-compatible API, M7 llm-auto orchestration). The M6 citation
         crawler returns references to the corpus, and every stage reads and
         writes a versioned SQLite/Arrow/Delta lakehouse substrate.
   :width: 100%
   :align: center

   Source markdown and JSON land in Delta tables, are tokenized into Arrow
   IPC shards, and are consumed by the Candle training loop -- every arrow
   crossing a versioned table boundary.

Performance Characteristics
---------------------------

On an Apple M1 Max (32 GB RAM):

.. list-table::
   :header-rows: 1

   * - Operation
     - Tokens/sec
     - Memory
   * - Inference (Q4_K)
     - 12–18
     - 6 GB
   * - Training (LoRA r=16)
     - 0.8–1.2
     - 22 GB
   * - Embedding (384-dim)
     - 150–200
     - 4 GB
   * - RAG search (HNSW)
     - 5000 qps
     - 1 GB

Compared to GPU-based training (A100 40GB):

- **Cost**: $0 vs $3.50/hour
- **Privacy**: 100% local vs cloud API
- **Throughput**: 10–50× slower, but sufficient for personal knowledge bases
