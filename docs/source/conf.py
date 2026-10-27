# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# The importable package root is <repo>/python/thotbook_ai/.
# The previous value ("../../../python") resolved outside the repo,
# which is why autodoc reported "No module named thotbook_ai".
sys.path.insert(0, os.path.abspath("../../python/thotbook_ai"))

# -- Project information -----------------------------------------------------
project = "thotbook-ai"
copyright = "2026, HFThot Research Lab"
author = "HFThot Research Lab"
release = "0.3.0"
version = "0.3.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "myst_parser",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The connectors in thotbook_ai import polars, pyarrow, requests and tokenizers
# at module scope. Those are runtime dependencies, not documentation ones, and
# installing them on the RTD builder would be slow and pointless, so autodoc
# imports stubs instead. Without this the automodule directives in
# api_reference.rst fail to import and the API page renders empty.
autodoc_mock_imports = [
    "polars",
    "pyarrow",
    "requests",
    "tokenizers",
]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = "thotbook-ai Documentation"
html_short_title = "thotbook-ai"
html_logo = "_static/logo_thot_transparent.png"
html_favicon = "_static/logo_thot_transparent.png"

# Announcement bar (top panel) - Furo supports this via html_theme_options
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#c2410c",
        "color-brand-content": "#c2410c",
        "color-api-pre-background": "#fef3e2",
        "color-background-primary": "#fefefe",
        "color-background-secondary": "#fef3e2",
        "color-foreground-primary": "#2d2d3a",
        "color-foreground-secondary": "#6b6b7a",
        "color-foreground-muted": "#9a9ab0",
        "color-border": "#fde68a",
        "color-announcement-background": "#c2410c",
        "color-announcement-text": "#ffffff",
    },
    "dark_css_variables": {
        "color-brand-primary": "#fb923c",
        "color-brand-content": "#fb923c",
        "color-api-pre-background": "#1a1a2e",
        "color-background-primary": "#141424",
        "color-background-secondary": "#1a1a2e",
        "color-foreground-primary": "#e8e8f0",
        "color-foreground-secondary": "#a0a0b8",
        "color-foreground-muted": "#707088",
        "color-border": "#3a250a",
        "color-announcement-background": "#fb923c",
        "color-announcement-text": "#ffffff",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/ThotDjehuty/thotbook-ai",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "announcement": "🚀 <strong>thotbook-ai v0.3.0</strong> — PINNs, federated learning &amp; privacy audit • <a href=\"https://hfthot-lab.eu/thotbook-ai.html\">HFThot Research Lab</a>",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/ThotDjehuty/thotbook-ai",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 24 24">
                    <path fill-rule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clip-rule="evenodd"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

# Napoleon settings for Google/NumPy docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# MyST parser configuration for markdown support
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Custom CSS — shared stylesheet with optimiz-rs
html_css_files = ["custom.css"]

# Mermaid configuration
mermaid_version = "10.9.0"
mermaid_init_js = (
    "mermaid.initialize({startOnLoad:true, theme:'dark', themeVariables:"
    "{primaryColor:'#c2410c',primaryTextColor:'#fff',primaryBorderColor:'#9a3412',"
    "lineColor:'#fb923c',secondaryColor:'#1a1a2e',tertiaryColor:'#0f0f1a'}});"
)

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True
copybutton_remove_prompts = True

# -- LaTeX / PDF ------------------------------------------------------------
# RTD builds a PDF as well as HTML. pdflatex aborts on glyphs outside the
# font encoding, so declare the ones our prose legitimately uses rather than
# letting the build fail on them.
latex_engine = "pdflatex"
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
    # pdflatex aborts on glyphs outside the font encoding. Declare the ones
    # our prose legitimately uses. These must be LITERAL characters --
    # backslash-u escapes are a Python notion, not a TeX one.
    "preamble": "\n".join([
        r"\usepackage{newunicodechar}",
        r"\newunicodechar{\u2192}{$\rightarrow$}".replace("\\u2192", "\u2192"),
        r"\newunicodechar{\u2265}{$\geq$}".replace("\\u2265", "\u2265"),
        r"\newunicodechar{\u2264}{$\leq$}".replace("\\u2264", "\u2264"),
        r"\newunicodechar{\u2248}{$\approx$}".replace("\\u2248", "\u2248"),
        r"\newunicodechar{\u00d7}{$\times$}".replace("\\u00d7", "\u00d7"),
        r"\newunicodechar{\u03b1}{$\alpha$}".replace("\\u03b1", "\u03b1"),
        r"\newunicodechar{\u03b2}{$\beta$}".replace("\\u03b2", "\u03b2"),
        r"\newunicodechar{\u03b8}{$\theta$}".replace("\\u03b8", "\u03b8"),
        r"\newunicodechar{\u03bb}{$\lambda$}".replace("\\u03bb", "\u03bb"),
        r"\newunicodechar{\u03bc}{$\mu$}".replace("\\u03bc", "\u03bc"),
        r"\newunicodechar{\u03c3}{$\sigma$}".replace("\\u03c3", "\u03c3"),
        r"\newunicodechar{\u2713}{$\checkmark$}".replace("\\u2713", "\u2713"),
    ]),
}

# Todo extension
todo_include_todos = True
todo_emit_warnings = True
