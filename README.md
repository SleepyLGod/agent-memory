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
- build an in-memory `DifferentiatedPolicy`
- run the current LOTUS-backed operator execution path

It does not yet:

- persist differentiated policies as durable JSON/YAML artifacts
- run optimizer passes
- persist views to storage

Runtime and adapter execution are intentionally in-memory: v0.0 can append rows,
execute compiled differentiated queries over the supported LOTUS adapter
operators, and query materialized views with compiled retrieval templates.
Unsupported expression operators such as arbitrary `filter(predicate=...)` and
`assign(...)` still raise explicit errors.

## Quick Start

```python
import agent_memory as am


spec = am.ClaudeMemory.spec()
print(sorted(spec.views))
policy = am.ClaudeMemory.differentiate_policy()
print(policy.view_execution_order)

memory = am.ClaudeMemory()
memory.add(am.Message(content="Please remember concise design docs."))
memory.query("design docs")
```

`ClaudeMemory` declares materialized views plus a parameterized
`retrieval_query = catalog.sem_topk(am.UserQuery(), 5)`. The in-memory
`DifferentiatedPolicy` stores differentiated view queries and retrieval query
templates; durable artifact IO and storage are still future work. Use the
HelloWorld smoke below for the minimal executable path.

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
  api.py                   Memory, Log, class-body spec/retrieval collection
  message.py               normalized append input object
  relation.py              DataFrame-style Relation and GroupedRelation operators
  logical.py               immutable QueryExpr, UserQuery, MemoryView, MemorySpec
  policy.py                DifferentialPolicyCompiler and DifferentiatedPolicy
  memories/claude.py       built-in ClaudeMemory policy
  planner/                 Q-to-Q' differential query planner interfaces
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
