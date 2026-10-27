API reference
=============

Python connectors
-----------------

The ``thotbook_ai`` package provides build-free connectors that shell out to
the compiled Rust binaries and read the resulting lakehouse with polars.

.. automodule:: thotbook_ai.corpus
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: thotbook_ai.redact
   :members:
   :undoc-members:

.. automodule:: thotbook_ai.tokenizer
   :members:
   :undoc-members:

.. automodule:: thotbook_ai.chat
   :members:
   :undoc-members:

HTTP surface (``llm-serve``)
----------------------------

.. list-table::
   :header-rows: 1

   * - Endpoint
     - Description
   * - ``GET /health``
     - Liveness probe
   * - ``GET /v1/models``
     - Model id (e.g. ``Qwen/Qwen3-8B``)
   * - ``POST /v1/chat/completions``
     - OpenAI-compatible chat completion

Delta tables (``datasets.*``)
-----------------------------

See :doc:`schemas` for the full column-by-column contract of every table.

* ``datasets.sessions`` — session transcripts (redacted)
* ``datasets.corpus`` — papers, notebooks and arXiv entries
* ``datasets.equations`` — LaTeX equations indexed per source
* ``datasets.citations`` — citation edges (seed set from the arXiv index)

Mathematical Operations
-----------------------

**Redaction Algorithm:**

The redaction operates on regex patterns:

.. math::

   R(d) = d \ominus \bigcup_{i=1}^n P_i

where :math:`P_i` are secret patterns and :math:`\ominus` denotes pattern replacement.

**Tokenization Mathematics:**

BPE merging maximizes frequency:

.. math::

   \text{merge}(a, b) = \arg\max_{(a,b) \in \mathcal{V}^2} \text{freq}(a, b)

**Chat Completion:**

The temperature-scaled sampling:

.. math::

   p(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}

where :math:`T` is the temperature and :math:`z_i` are logits.

**Lakehouse Operations:**

Append with versioning:

.. math::

   \text{version}_{n+1} = \text{version}_n + 1, \qquad
   \Delta_{n+1} = \text{append}(\Delta_n, \text{new\_data})

* ``datasets.arxiv_text`` — M6 crawler output (metadata + full text + edges)
