# agent-memory

`agent-memory` is an early semantic memory framework for agent developers.
The design goal is to help agent applications maintain durable memory over a
growing log of messages and events.

The current design centers on four ideas:

- `Log`: the append-only source of messages or events.
- `Views`: materialized semantic memory derived from the log or other views.
- `Semantic operators`: operations such as grouping, mapping, filtering, and ranking that may compile to optimizable LM programs.
- `Stores`: physical materialization choices for logs and views.

DSPy may be used underneath as a compilation and optimization backend for
semantic operators. It is not the product identity, and the public API is still
under active design.

## Status

This repository is in design/prototype mode. The current source code should be
treated as experimental and may be deleted or rewritten as the interface
stabilizes. Do not treat the runtime prototype as the API contract.

The main current design note is:

```text
docs/design/user-interface-design.md
```

## Environment

Install and sync dependencies with `uv`:

```bash
uv sync
```

Register the local Jupyter kernel if notebook exploration is needed:

```bash
uv run python -m ipykernel install --user --name agent-memory --display-name "Python (agent-memory)"
```
