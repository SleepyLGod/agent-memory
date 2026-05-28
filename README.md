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
- inspect logical `QueryExpr` trees
- instantiate the built-in `ClaudeMemory` policy
- run the minimal LOTUS e2e path for row-local `sem_filter`, `sem_map`, and `sem_topk`

It does not yet:

- run general differential rules beyond row-local `sem_filter` and `sem_map`
- execute general semantic view maintenance beyond the current narrow LOTUS path
- persist views to storage

Runtime and adapter execution are intentionally narrow: v0.0 can append rows,
maintain row-local `sem_filter`/`sem_map` views, and query materialized views
with `sem_topk`. Unsupported policies still raise `NotImplementedError`.

## Quick Start

```python
import agent_memory as am


spec = am.ClaudeMemory.spec()
print(sorted(spec.views))

memory = am.ClaudeMemory()
memory.add(am.Message(content="Please remember concise design docs."))
memory.query("design docs")
```

`ClaudeMemory` includes operators beyond the current row-local runtime subset,
so `add(...)` and `query(...)` still raise explicit errors. Use the HelloWorld
smoke below for the minimal executable path.

For an interface-only smoke demo:

```bash
uv run examples/claude/interface_smoke.py
```

For a real LOTUS-backed HelloWorld e2e over a small LOCOMO dialogue slice, copy
`.env.example` to `.env`, set `DEEPSEEK_API_KEY`, then run:

```bash
uv run examples/helloworld/helloworld_smoke.py
```

The default LOTUS model is `deepseek/deepseek-v4-pro`. Advanced users can
override it by passing a custom `LotusAdapter(model=...)`.


To include the same real LOTUS path in pytest, set
`AGENT_MEMORY_RUN_LOTUS_E2E=1`. The default test suite does not call external
model APIs.

## V0.0 File Structure

```text
src/agent_memory/
  __init__.py              public package exports
  api.py                   Memory, Log, class-body spec collection, query wrapper
  message.py               normalized append input object
  relation.py              DataFrame-style Relation and GroupedRelation operators
  logical.py               immutable QueryExpr, MemoryView, MemorySpec
  memories/claude.py       built-in ClaudeMemory policy
  planner/                 future Q-to-ΔQ differential query planner interfaces
  runtime/                 runtime state and future execution shell
  adapters/                execution adapter protocol and LotusAdapter shell

examples/claude/
  interface_smoke.py       inspectable v0.0 interface demo

examples/helloworld/
  helloworld_smoke.py      real LOCOMO + LOTUS sem_filter/sem_map/top-k smoke
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

Run the real HelloWorld smoke:

```bash
uv run examples/helloworld/helloworld_smoke.py
```

Run tests:

```bash
uv run --with pytest python -m pytest
```

Run tests including the real LOTUS e2e:

```bash
AGENT_MEMORY_RUN_LOTUS_E2E=1 uv run --with pytest python -m pytest
```

Register the local Jupyter kernel if notebook exploration is needed:

```bash
uv run python -m ipykernel install --user --name agent-memory --display-name "Python (agent-memory)"
```
