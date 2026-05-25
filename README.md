# agent-memory

`agent-memory` is an experimental semantic memory framework for agent
developers. The v0.0 direction is DataFrame-first: policy authors describe
memory as logical views over an append-only log, and the system will later
rewrite and maintain those views incrementally.

## Status

The current code implements the v0.0 interface layer.

It can:

- import `agent_memory as am`
- define DataFrame-style memory policies
- collect a `MemorySpec` from a policy class
- inspect logical `RelationExpr` trees
- instantiate the built-in `ClaudeMemory` policy

It does not yet:

- ingest log rows
- execute `add(...)`
- execute `query(...)`
- run the maintenance planner
- execute LOTUS or model calls
- persist views to storage

Runtime and adapter execution are not implemented yet; methods that would run
real maintenance or retrieval raise `NotImplementedError`.

## Quick Start

```python
import agent_memory as am


spec = am.ClaudeMemory.spec()
print(sorted(spec.views))

memory = am.ClaudeMemory()
memory.add(am.Message(content="Please remember concise design docs."))
memory.query("design docs")
```

In the current interface-only stage, the final two calls raise explicit
`NotImplementedError`s. The point is to validate the v0.0 API and logical plan
construction, not runtime execution.

For a runnable smoke demo:

```bash
uv run examples/claude/interface_smoke.py
```

## V0.0 File Structure

```text
src/agent_memory/
  __init__.py              public package exports
  api.py                   Memory, Log, class-body spec collection, query wrapper
  message.py               normalized append input object
  relation.py              DataFrame-style Relation and GroupedRelation operators
  logical.py               immutable RelationExpr, MemoryView, MemorySpec
  memories/claude.py       built-in ClaudeMemory policy
  planner/                 future Q-to-Q' maintenance planner interfaces
  runtime/                 runtime state and future execution shell
  adapters/                execution adapter protocol and LotusAdapter shell

examples/claude/
  interface_smoke.py       inspectable v0.0 interface demo
```

## Design Documents

- `docs/design/v0.0-interface.md`: current v0.0 interface boundary
- `docs/design/operator-api.md`: DataFrame-style semantic operator API
- `docs/design/groupby-aggregation-api-rationale.md`: chain aggregation rationale
- `docs/design/TODO.md`: unresolved v0.0 -> v0.1 interface and research questions
- `docs/optimization/incremental-semantic-view-maintenance.tex`: semantic IVM theory draft

## Development

Install and sync dependencies with `uv`:

```bash
uv sync
```

Run the interface smoke demo:

```bash
uv run examples/claude/interface_smoke.py
```

Run tests:

```bash
uv run --with pytest python -m pytest
```

Register the local Jupyter kernel if notebook exploration is needed:

```bash
uv run python -m ipykernel install --user --name agent-memory --display-name "Python (agent-memory)"
```
