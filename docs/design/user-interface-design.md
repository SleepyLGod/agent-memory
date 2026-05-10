# User Interface Design

This document records the current public interface direction. It is a design target, not a promise that every API below is implemented in the current prototype.

## Policy Authoring

Policy writers define typed schemas with normal Python/Pydantic objects. A schema does not need to inherit from `am.Node`.

```python
from __future__ import annotations

from pydantic import BaseModel, Field

import agent_memory as am


class Topic(BaseModel):
    """Durable topic memory."""

    title: str = Field(description="stable topic title")
    description: str = Field(description="short summary used for retrieval")
    content: str = Field(description="full durable memory content")


class CatalogEntry(BaseModel):
    """A compact catalog entry for topic memory."""

    title: str = Field(description="topic title shown in the catalog")
    topic_id: str = Field(description="stable identifier of the topic view item")
    hook: str = Field(description="one-line retrieval hint")


class ClaudeMemory(am.Memory):
    """Claude-style long-term memory over an append-only log."""

    # `log` is defined by the memory runtime.
    # It represents messages/events passed to memory.add(...).
    log = am.Log()

    topic = log.sem_groupby(
        Topic,
        "Group durable user and project memories into stable topic buckets. "
        "Skip transient task details. Update existing topics when appropriate.",
    ).refresh(on="turn")

    # Catalog is a normal materialized view.
    catalog = topic.sem_map(
        CatalogEntry,
        "Create compact catalog entries for topic memories.",
    ).refresh(on="source")

    def query(self, query: str, *, k: int = 5):
        return self.topic.sem_topk(
            query,
            "Select the most relevant topic memories for the query.",
            k=k,
            context=self.catalog,
        )
```

Key points:

- `Topic` and `CatalogEntry` are ordinary typed schemas.
- `log` is the reserved append-only input relation for `memory.add(...)`.
- `topic` and `catalog` are logical materialized views.
- `catalog` is used as explicit operator context in `query(...)`.
- Semantic operators can lower to DSPy modules, but the user interface does not require users to write DSPy code.

## Conceptual Model

The interface separates logical memory definitions from physical maintenance.

- `Log` is an append-only source relation for messages and events.
- A view query is a semantic path from one relation or view to a target typed schema, such as `log.sem_groupby(Topic, ...)`.
- `window(...)` changes the data range or grouping seen by the query. It is not a refresh trigger.
- `refresh(...)` defines when a materialized memory view is maintained. It is not a semantic operator.
- Store bindings define where log rows and materialized view rows are persisted. They are separate from schema, query, and refresh policy.

This mirrors existing data-system concepts but keeps the authoring surface Pythonic. In PostgreSQL, a materialized view is defined by `AS query` and later refreshed. In Flink Materialized Table, a table has a query definition plus refresh mode and freshness. In Flink DataStream, window assignment and trigger/fire behavior are separate concepts. `agent-memory` follows the same separation: query path, windowing, refresh timing, and storage are distinct layers.

### View Query And Refresh

Semantic operators remain general-purpose operators:

```python
view.sem_map(TargetSchema, "...")
view.sem_filter("...")
view.sem_groupby(TargetSchema, "...")
view.sem_join(TargetSchema, "...", context=other_view)
view.sem_topk(query, "...", k=5, context=catalog)
view.sem_window("...")
```

The user-facing API should not introduce special grouped-upsert semantic operators. Runtime implementations may compile a `sem_groupby(...)` view query into extraction, matching, merge, update, and storage operations, but those are physical maintenance details.

`refresh(...)` is the fluent timing policy for maintaining a materialized view:

```python
topic = log.sem_groupby(Topic, "...").refresh(on="turn")
catalog = topic.sem_map(CatalogEntry, "...").refresh(on="source")
summary = topic.sem_map(Summary, "...").refresh(on="query")
profile = topic.sem_groupby(Profile, "...").refresh(every="24h")
profile = topic.sem_groupby(Profile, "...").refresh(every=3, unit="turn")
profile = topic.sem_groupby(Profile, "...").refresh(every="24h", require={"sessions": 5})
draft = log.sem_map(Draft, "...").refresh(mode="manual")
```

Supported refresh concepts in the design target:

- `on="turn"`: maintain after an agent turn.
- `on="source"`: maintain after the upstream source or view changes.
- `on="query"`: lazily refresh when queried.
- `every="24h"`: time-based periodic refresh.
- `every=3, unit="turn"`: count-based refresh over turns, messages, sessions, or another supported unit.
- `require={...}`: additional gates that must be satisfied before refresh runs.
- `mode="manual"`: refresh only when explicitly invoked by the runtime or user.

Inside an `am.Memory` class body, assignments are declarative rule assignments. The first assignment to a view name defines the primary view query. A repeated assignment to the same view name appends a self-maintenance rule for the same logical view instead of ordinary Python overwrite. The implementation will need a metaclass or custom class namespace to capture assignment history and validate that repeated assignments target the same logical schema.

## Direct Use

```python
memory = ClaudeMemory(
    stores={
        "log": am.stores.JSONL(".memory/log.jsonl"),
        "topic": am.stores.Directory(".memory/topics"),
        "catalog": am.stores.Markdown(".memory/MEMORY.md"),
    }
)

memory.add([
    {
        "role": "user",
        "content": "I prefer concise design docs with clear tradeoffs.",
    }
])

results = memory.query("How should I write design docs?")
```

## Store

`Store` describes physical materialization. It is not the schema, not the view definition, and not the workflow. The same logical memory policy should be able to use different stores without changing the schema or semantic operators.

`Log` is the append-only event/message source. It is also materialized by a store, for example a JSONL file, SQLite table, or external database table.

Each logical view can bind to one or more stores. This includes `catalog`. `catalog` is just a normal materialized view. It can be generated by semantic operators, maintained like other views, and passed explicitly as operator context during retrieval.

Conceptual store examples:

- `am.stores.JSONL(...)` for append-only logs.
- `am.stores.Directory(...)` for one-file-per-record materialization.
- `am.stores.Markdown(...)` for Markdown materialization such as topic files or catalog views.
- `am.stores.SQLite(...)` / `am.stores.Postgres(...)` for relational durable storage.
- Vector stores for embedding-backed retrieval materialization.
- Graph stores for relationship-oriented materialization.

Store binding example:

```python
memory = ClaudeMemory(
    stores={
        "log": am.stores.JSONL(".memory/log.jsonl"),
        "topic": am.stores.Directory(".memory/topics"),
        "catalog": am.stores.Markdown(".memory/MEMORY.md"),
    }
)
```

Store binding rules:

- Store keys match logical view names, or the reserved key `"log"`.
- A value can be one store or an ordered list of stores.
- If multiple stores are listed, order is the simple v0 convention for how writes are attempted or mirrored.
- Store choices are separate from the logical schema and view definitions.

## Built-In Memory

For common use, users should not have to write a policy class:

```python
import agent_memory as am

memory = am.ClaudeMemory()
memory.add("I prefer concise design docs.")
results = memory.query("design docs")
```

The built-in class can choose default stores internally. Users can override stores only when they need a specific local or external persistence layout.

## Agent Framework Integration

AutoGen-style usage should use the framework's native memory slot:

```python
from autogen_agentchat.agents import AssistantAgent
import agent_memory as am

memory = am.ClaudeMemory()

assistant = AssistantAgent(
    name="assistant",
    model_client=model_client,
    memory=[memory],
)
```

LangGraph-style usage should expose a store adapter:

```python
import agent_memory as am

memory = am.ClaudeMemory()
graph = builder.compile(store=memory.as_langgraph_store())
```

The adapter layer should follow each agent framework's native API instead of forcing users to call a separate memory protocol inside the agent loop.

## Window

Window is an operator on view-like relations. It can apply to `am.Log()`, materialized views, or intermediate semantic views.

`window(...)` is about data scope. `refresh(...)` is about when a materialized view is maintained. A query may use both, but they are different layers.

Deterministic window:

```python
view.window(count=20)
view.window(time="7d")
view.window(size=100, step=100, unit="count")   # tumbling count
view.window(size=100, step=10, unit="count")    # sliding count
view.window(size="1h", step="1h", unit="time")  # tumbling time
view.window(size="1h", step="5m", unit="time")  # sliding time
```

`view.window(...)` is deterministic count/time windowing. It does not call an LLM.

Semantic window:

```python
view.sem_window(
    "Split this view into coherent memory-worthy windows."
)
```

`view.sem_window(...)` is semantic windowing / segmentation. It lowers to a semantic operator and can compile to a DSPy module.

Optional sugar, not the primary API:

```python
view.window.tumbling(time="1h")
view.window.sliding(time="1h", step="5m")
view.window.count(size=100, step=10)
```
