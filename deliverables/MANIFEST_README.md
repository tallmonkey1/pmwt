# modules.json -- detailed dependency manifest (schema v2.0)

Auto-extracted from the source code by extract_deps.py. Read this file to understand
every module's responsibility, public surface, and the precise symbols that flow
across every internal and external dependency edge.

## Top-level structure

| Key | Content |
|---|---|
| schema_version | Always "2.0" for this format. |
| project | Project metadata (name, version, design principles). |
| totals | Counts and a verified DAG-flag. package_level_acyclic: true means the package graph has no cycles (verified by automated DFS). |
| package_dependency_layers | Human-readable layer assignment for each package. |
| packages | Per-package summary: responsibility, layer, module list. |
| package_edges | Edges at the package level with the count of underlying module edges. |
| package_fan_out / package_fan_in | Adjacency lists for graph visualisation. |
| external_dependencies | **Per-external-package**: module name, *purpose*, and the full list of imported symbols (imported_symbols). |
| modules | **Per source module**: see below. |
| internal_edges | **Per internal edge**: from, to, and the full list of imported symbols (symbols). |

## Per-module record (the meat of the manifest)

Every module exports a single record with these keys:

| Key | Content |
|---|---|
| module | Bare dotted module path (no options_engine. prefix). |
| package | The package this module belongs to. |
| layer | The topological layer (foundation, quantitative_core, calibration_layer, inference_layers, decision_layer, learning_layer, evaluation_layer, execution_layer). |
| path | File path relative to the repo root. |
| docstring_summary | First non-blank line of the module docstring. |
| docstring_full | The full module docstring. |
| loc | Source lines of code. |
| constants | Module-level UPPER_CASE assignments (name + unparsed value). |
| public_api | List of public-name records (see below). |
| validation_rules | List of validation / invariant rules extracted from the docstring. |
| internal_dependencies | Per-target: {module, symbols: [...]}. |
| external_dependencies | Per-target: {module, purpose, imported_symbols: [...]}. |
| criticality | **Importance ranking** for simulation accuracy, training accuracy, regression sensitivity, and fallback strategy (see `CRITICALITY.md`). |

## Per public-name record

Classes:
* kind = "class"
* bases, decorators
* fields: {name, annotation, default}
* methods: {name, kind, summary, signature, raises}
* properties: same as methods but is_property = true

Functions / methods:
* kind = "function" or "method"
* summary: first non-blank line of the docstring
* signature: positional, keyword_only, vararg, kwarg, return_annotation, decorators
* raises: list of exception classes / conditions extracted from the docstring's Raises section

Constants:
* kind = "constant", annotation, default

## How to read the manifest

* **Find what a module does**: read its docstring_summary and validation_rules.
* **Find what it exports**: read public_api (classes first, then functions, then constants).
* **Find its inputs and outputs**: every public name has a signature with parameters (name / annotation / default) and return_annotation.
* **Find what exceptions it raises**: the raises list on each public name.
* **Find what it depends on**: internal_dependencies and external_dependencies list the **specific symbols** imported from each target module.
* **Trace data flow across the graph**: every internal_edges entry has a symbols list -- the precise names that flow from one module to another.

## Totals (current run)


* source modules: 103
* internal edges: 402 (1126 symbol imports tracked)
* external deps: 25
* packages: 16
* package edges: 33
* package-level DAG verified acyclic: True
