thotbook-ai Documentation
=============================

**A private, local-first LLM training-to-serving pipeline in Rust — free by
design, no API key, nothing leaves your machine.**

.. only:: html

   .. image:: https://img.shields.io/badge/version-0.3.0-blue.svg
      :target: https://hfthot-lab.eu/thotbook-ai.html
      :alt: Version

   .. image:: https://img.shields.io/badge/license-Apache--2.0-green.svg
      :target: https://hfthot-lab.eu/thotbook-ai.html
      :alt: License

   .. image:: https://img.shields.io/badge/Rust-1.92+-orange.svg
      :alt: Rust 1.92+

   .. image:: https://img.shields.io/badge/CPU_only-yes-purple.svg
      :alt: CPU-only

thotbook-ai turns your private documents, research notes, session
transcripts and paper indexes into a *personal language model* — fully on
your own hardware. Six focused Rust crates are wired together by a Polarway
Delta lakehouse and exposed through an OpenAI-compatible HTTP surface, so any
existing tool (opencode, Jupyter, curl) can talk to it — free, private and
open-source.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started
   architecture
   schemas

.. toctree::
   :maxdepth: 2
   :caption: Algorithms

   algorithms
   algorithms/training
   algorithms/inference
   algorithms/embeddings
   algorithms/rag
   algorithms/orchestration

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api_reference
   roadmap

Features
--------

 **Private by design** — content is redacted at ingest time; a write-time
leak audit refuses any row that still contains secrets after redaction.

 **No API key, no billing** — a fully self-hosted Qwen3-8B server
(`llm-serve`) speaking the OpenAI-compatible protocol, free for a reason.

 **CPU-only** — quantization and LoRA keep training and inference feasible
on a regular workstation (32 GB RAM recommended).

 **Lakehouse-native** — every stage produces versioned, queryable Delta
tables (`datasets.*`) with time-travel.

 **Parity everywhere** — redaction rules and tokenizer defaults stay in
lock-step between Rust, Python and the JS reference.

Quick example
-------------

.. code-block:: python

    from thotbook_ai import ChatClient

    with ChatClient() as assistant:          # http://127.0.0.1:8100/v1
        if assistant.is_online():
            reply = assistant.ask("Give one line of Kähler geometry intuition.")
            print(reply.text)

Run the same model from the shell:

.. code-block:: sh

    python -m thotbook_ai chat "Explain a fibration in one line." --max-tokens 64

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`