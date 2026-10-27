Roadmap
=======

In development
--------------

* **M3 ``llm-train``** — CPU QLoRA SFT of Qwen3-4B on the tokenized corpus,
  reusing the LoRA / Q4-quantization math in ``optimiz-rs``.
* **M4 ``llm-rag``** — corpus embeddings + HNSW retrieval (the HNSW index is
  already available as a reusable primitive in ``optimiz-rs``).
* **``llm-cli``** — a single entrypoint orchestrating the whole pipeline.

Planned
-------

* **M6 citation crawler** — arXiv reference-closure crawl to enrich the
  ``datasets.citations`` edges (arXiv API + OpenAlex / Semantic Scholar).
* **Time-travel audit tooling** — inspect historical versions of any
  ``datasets.*`` table through the Polarway lakehouse.

Mathematical Foundations for Future Milestones
----------------------------------------------

**M3: QLoRA Training Mathematics**

The QLoRA method combines quantization with low-rank adaptation:

.. math::

   \text{Quantize}(W_0) = \text{round}\left(\frac{W_0}{\Delta}\right) + \text{dequantize}

where :math:`\Delta = \frac{\max(|W_0|)}{2^k - 1}` for k-bit quantization.

The gradient update for LoRA parameters:

.. math::

   \Delta B = \eta \frac{\partial \mathcal{L}}{\partial B}, \qquad \Delta A = \eta \frac{\partial \mathcal{L}}{\partial A}

with learning rate :math:`\eta` and loss :math:`\mathcal{L}`.

**M4: HNSW Retrieval Theory**

The HNSW graph construction connects each node to :math:`M` neighbors at each
layer. Search complexity is:

.. math::

   \mathcal{O}(\log_{M} N) \text{ hops}, \qquad \text{accuracy} \approx 1 - e^{-c\sqrt{\log N}}

where :math:`c` is a constant depending on :math:`M` and the data distribution.

**M6: Citation Graph Theory**

The citation graph is modeled as a directed acyclic graph (DAG):

.. math::

   G = (V, E), \qquad E = \{(u, v) : u \text{ cites } v\}

PageRank-style authority scores:

.. math::

   \text{PR}(v) = \frac{1-d}{|V|} + d \sum_{u \in \mathcal{B}(v)} \frac{\text{PR}(u)}{|\mathcal{F}(u)|}

where :math:`d` is the damping factor, :math:`\mathcal{B}(v)` are backlinks,
and :math:`\mathcal{F}(u)` are forward links.
