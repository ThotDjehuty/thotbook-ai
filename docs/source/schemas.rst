Schemas
=======

This page is the machine-readable contract of the whole pipeline: the Delta
lakehouse table schemas written by ``llm-corpus``, the OpenAI-compatible HTTP
surface of ``llm-serve``, and the environment configuration it honours.
Schemas are the single source of truth for any tool that reads a ``datasets.*``
table or calls the model server.

.. contents:: On this page
   :local:

Delta lakehouse table schemas
-----------------------------

Every stage reads from / writes to versioned Delta tables under
``data/lakehouse``. Columns are exact mirrors of
``crates/llm-corpus/src/schema.rs`` (``StructField`` definitions). ``N/A``
marks columns that are nullable.

``datasets.sessions``
~~~~~~~~~~~~~~~~~~~~~

One row per persisted session document (from ``historia/``).

.. list-table::
   :header-rows: 1
   :widths: 22 12 14 52

   * - Column
     - Type
     - Nullable
     - Description
   * - ``id``
     - string
     - no
     - Session identifier
   * - ``filename``
     - string
     - no
     - ``YYYYMMDD_<slug>.md`` document name
   * - ``date``
     - string
     - no
     - Session date (``YYYY-MM-DD``)
   * - ``title``
     - string
     - no
     - Document title
   * - ``content``
     - string
     - no
     - Redacted transcript body
   * - ``tags``
     - string
     - no
     - Comma separated tags
   * - ``category``
     - string
     - no
     - Historia category
   * - ``word_count``
     - long
     - no
     - Tokenizable word estimate
   * - ``indexed_at``
     - string
     - no
     - UTC timestamp of the write
   * - ``file_modified``
     - string
     - yes
     - Source file mtime, if known

``datasets.corpus``
~~~~~~~~~~~~~~~~~~~

Unified research corpus (papers + notebooks + arXiv entries).

.. list-table::
   :header-rows: 1
   :widths: 22 12 14 52

   * - Column
     - Type
     - Nullable
     - Description
   * - ``source_type``
     - string
     - no
     - ``paper`` | ``notebook`` | ``arxiv``
   * - ``source_id``
     - string
     - no
     - Stable id inside the source type
   * - ``title``
     - string
     - no
     - Document title
   * - ``authors``
     - string
     - no
     - Comma separated authors
   * - ``content``
     - string
     - no
     - Redacted full text / markdown
   * - ``tags``
     - string
     - no
     - Comma separated tags
   * - ``category``
     - string
     - no
     - Research category
   * - ``year``
     - long
     - yes
     - Publication year if known
   * - ``word_count``
     - long
     - no
     - Word estimate of ``content``
   * - ``equation_count``
     - long
     - no
     - Number of extracted LaTeX equations
   * - ``indexed_at``
     - string
     - no
     - UTC timestamp of the write

``datasets.equations``
~~~~~~~~~~~~~~~~~~~~~~

LaTeX equations extracted per source section.

.. list-table::
   :header-rows: 1
   :widths: 22 12 14 52

   * - Column
     - Type
     - Nullable
     - Description
   * - ``id``
     - long
     - no
     - Monotonic equation id
   * - ``latex``
     - string
     - no
     - Raw LaTeX body
   * - ``label``
     - string
     - no
     - LaTeX ``\\label`` value
   * - ``context``
     - string
     - no
     - Surrounding sentence
   * - ``source_type``
     - string
     - no
     - Originating source type
   * - ``source_id``
     - string
     - no
     - Originating source id
   * - ``section``
     - string
     - no
     - Section number / name
   * - ``indexed_at``
     - string
     - no
     - UTC timestamp of the write

``datasets.citations``
~~~~~~~~~~~~~~~~~~~~~~

arXiv metadata + reference-closure edges (seed set).

.. list-table::
   :header-rows: 1
   :widths: 22 12 14 52

   * - Column
     - Type
     - Nullable
     - Description
   * - ``arxiv_id``
     - string
     - no
     - arXiv identifier
   * - ``title``
     - string
     - no
     - Paper title
   * - ``authors``
     - string
     - no
     - Comma separated authors
   * - ``categories``
     - string
     - no
     - Comma separated category codes
   * - ``primary_category``
     - string
     - no
     - Primary category code
   * - ``published_date``
     - string
     - no
     - ``YYYY-MM-DD``
   * - ``abstract``
     - string
     - no
     - Abstract text
   * - ``cited_ids``
     - string
     - no
     - Whitespace separated cited arXiv ids
   * - ``fetch_status``
     - string
     - no
     - ``pending`` | ``fetched`` | ``failed``
   * - ``indexed_at``
     - string
     - no
     - UTC timestamp of the write

``datasets.arxiv_text``
~~~~~~~~~~~~~~~~~~~~~~~

M6 crawler output: metadata + full text (ar5iv) + reference-closure edges.

.. list-table::
   :header-rows: 1
   :widths: 30 12 12 46

   * - Column
     - Type
     - Nullable
     - Description
   * - ``arxiv_id``
     - string
     - no
     - arXiv identifier
   * - ``title`` / ``authors``
     - string
     - no
     - As in ``datasets.citations``
   * - ``categories`` / ``primary_category``
     - string
     - no
     - As in ``datasets.citations``
   * - ``published_date``
     - string
     - no
     - ``YYYY-MM-DD``
   * - ``abstract``
     - string
     - no
     - Abstract text
   * - ``full_text``
     - string
     - no
     - ar5iv extracted full text
   * - ``cited_ids``
     - string
     - no
     - Whitespace separated cited arXiv ids
   * - ``fetch_status``
     - string
     - no
     - ``pending`` | ``fetched`` | ``failed``
   * - ``indexed_at``
     - string
     - no
     - UTC timestamp of the write

HTTP surface (``llm-serve``)
----------------------------

.. list-table::
   :header-rows: 1

   * - Endpoint
     - Method
     - Description
   * - ``/health``
     - ``GET``
     - Liveness probe
   * - ``/v1/models``
     - ``GET``
     - Served model ids
   * - ``/v1/chat/completions``
     - ``POST``
     - OpenAI-compatible chat completion

``GET /health``
~~~~~~~~~~~~~~~

.. code-block:: json

   {"status": "ok"}

``GET /v1/models``
~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "object": "list",
     "data": [
       {"id": "Qwen/Qwen3-8B", "object": "model", "owned_by": "local"}
     ]
   }

``POST /v1/chat/completions`` — request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All fields except ``messages`` are optional (the server applies ``LLM_SERVE_*``
defaults).

.. code-block:: json

   {
     "model": "Qwen/Qwen3-8B",
     "messages": [
       {"role": "system", "content": "You are a mathematician."},
       {"role": "user", "content": "Explain a fibration in one line."}
     ],
     "max_tokens": 512,
     "temperature": 0.7,
     "top_p": 0.9
   }

Roles
~~~~~

.. list-table::
   :header-rows: 1

   * - Role
     - Meaning
   * - ``user``
     - A user turn (always present)
   * - ``system``
     - Optional system instruction (rendered first)
   * - ``assistant``
     - Optional prior model turn (continuation)

``POST /v1/chat/completions`` — response
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "id": "chatcmpl-0000000000000000",
     "object": "chat.completion",
     "created": 1738000000,
     "model": "Qwen/Qwen3-8B",
     "choices": [
       {
         "index": 0,
         "message": {"role": "assistant", "content": "…"},
         "finish_reason": "stop"
       }
     ]
   }

Error body (HTTP ``400`` / ``500``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "error": {
       "message": "error message",
       "type": "error",
       "param": null,
       "code": "error"
     }
   }

Environment configuration (``LLM_SERVE_*``)
-------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 10 56

   * - Variable
     - Default
     - Description
   * - ``LLM_SERVE_HOST``
     - ``127.0.0.1``
     - Bind address
   * - ``LLM_SERVE_PORT``
     - ``8100``
     - Bind port
   * - ``LLM_SERVE_MODEL_DIR``
     - ``data/models/qwen3-8b``
     - Candle weights directory
   * - ``LLM_SERVE_MODEL_ID``
     - ``Qwen/Qwen3-8B``
     - Id reported by ``/v1/models``
   * - ``LLM_SERVE_MAX_TOKENS``
     - ``512``
     - Default ``max_tokens``
   * - ``LLM_SERVE_TEMPERATURE``
     - ``0.7``
     - Default sampling temperature
   * - ``LLM_SERVE_TOP_P``
     - ``0.9``
     - Default nucleus sampling

Mathematical Models
-------------------

**Token Probability Distribution:**

For a vocabulary :math:`\mathcal{V}` of size :math:`|\mathcal{V}|`, the
probability of token :math:`x_t` given context :math:`x_{<t}`:

.. math::

   p(x_t | x_{<t}) = \text{softmax}(W_h h_t + W_x x_t + b)

**Redaction Pattern Matching:**

The redaction operator :math:`R` applies pattern set :math:`\mathcal{P}`:

.. math::

   R(d) = \bigotimes_{p \in \mathcal{P}} \text{replace}(d, p, \text{[REDACTED]})

where :math:`\bigotimes` denotes sequential application.

**Lakehouse Versioning:**

Time-travel query at version :math:`v`:

.. math::

   \text{query}(v) = \text{scan}(\text{table}, \text{snapshot}(v))

**Temperature Scaling:**

The temperature :math:`T` controls sampling diversity:

.. math::

   p_T(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}

As :math:`T \to 0`, sampling becomes greedy; as :math:`T \to \infty`, uniform.