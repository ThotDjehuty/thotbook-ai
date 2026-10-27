Getting started
===============

Requirements
------------

* Rust 1.92+ (stable)
* Python 3.10+ (optional, for the Python connectors)
* 16 GB+ RAM (32 GB recommended)

The workspace resolves ``polarway-lakehouse`` (a private Delta-lakehouse
companion crate) from a sibling checkout via path dependency
(``../polarway/polarway-lakehouse``). Check it out next to this repository.

Build
-----

.. code-block:: sh

   cargo build --release

Ingest
------

.. code-block:: sh

   cargo run --release -p llm-corpus -- ingest \
     --historia  /path/to/historia.db \
     --thotbook  /path/to/thotbook.db \
     --lakehouse data/lakehouse

Writes ``datasets.sessions``, ``datasets.corpus``, ``datasets.equations`` and
``datasets.citations``. Rows that still contain secrets after redaction are
reported and the write is refused.

Tokenize
--------

.. code-block:: sh

   cargo run --release -p llm-tokenize -- tokenize \
     --lakehouse data/lakehouse \
     --table datasets.sessions

The Qwen3 tokenizer is downloaded on first use to
``data/tokenizers/qwen3-tokenizer.json``.

Serve
-----

.. code-block:: sh

   cargo run --release -p llm-serve -- --model-dir data/models/qwen3-8b
   # OpenAI-compatible server on http://127.0.0.1:8100

Configuration is env-driven (``LLM_SERVE_*``) with CLI overrides:

* ``LLM_SERVE_HOST`` (default ``127.0.0.1``)
* ``LLM_SERVE_PORT`` (default ``8100``)
* ``LLM_SERVE_MODEL_DIR`` (default ``data/models/qwen3-8b``)
* ``LLM_SERVE_MODEL_ID`` (default ``Qwen/Qwen3-8B``)
* ``LLM_SERVE_MAX_TOKENS`` (default ``512``)
* ``LLM_SERVE_TEMPERATURE`` (default ``0.7``)
* ``LLM_SERVE_TOP_P`` (default ``0.9``)

Chat with the local model
-------------------------

The Python connectors ship a ``ChatClient`` that speaks the OpenAI-compatible
surface — no API key, no billing, everything stays on your machine:

.. code-block:: python

   from thotbook_ai import ChatClient

   with ChatClient() as assistant:      # http://127.0.0.1:8100/v1
       if assistant.is_online():
           reply = assistant.ask("Give one line of Kähler geometry intuition.")
           print(reply.text)

or from the shell:

.. code-block:: sh

   python -m thotbook_ai chat "Explain a fibration in one line." --max-tokens 64
   python -m thotbook_ai health

Testing
-------

.. code-block:: sh

   cargo test --workspace   # 39 tests across llm-corpus / llm-tokenize / llm-serve
   pytest python/thotbook_ai/tests   # ChatClient mock + live integration (auto-skipped when offline)

Mathematical Foundations
------------------------

The thotbook-ai pipeline is grounded in established mathematical principles:

**Redaction as Information Projection:**

The redaction step projects sensitive information onto a safe subspace. Let
:math:`\mathcal{S}` be the set of secrets and :math:`\mathcal{D}` the document space.
The redaction operator :math:`R: \mathcal{D} \to \mathcal{D}` satisfies:

.. math::

   R(d) = d \setminus \mathcal{S}, \qquad R(R(d)) = R(d) \text{ (idempotence)}

**Tokenization as Discrete Embedding:**

The tokenizer maps continuous text to discrete tokens:

.. math::

   \text{tok}: \mathcal{V}^* \to \{0, 1, \ldots, |V|-1\}^n

where :math:`\mathcal{V}` is the vocabulary and :math:`n` is the sequence length.

**Lakehouse as Versioned State:**

The Delta lakehouse provides ACID transactions with time-travel:

.. math::

   \mathcal{L}_t = \{\text{tables at version } t\}, \qquad \mathcal{L}_t \to \mathcal{L}_{t+1} \text{ via append operations}

**Inference as Conditional Generation:**

The model server implements:

.. math::

   p(x_{t+1} | x_{\leq t}) = \text{softmax}(W_h h_t + W_x x_t)

where :math:`h_t` is the hidden state from the transformer backbone.
