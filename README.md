# dspy-memory

`dspy-memory` is an early research package for designing DSPy-compatible memory
policies. The current goal is to explore how memory schemas, workflows, and
semantic operators can map cleanly onto DSPy-style signatures, modules, and
optimizers.

This repository is intentionally minimal right now. It contains the Python
package skeleton, a `uv` environment, DSPy dependencies, and notebook support.
The concrete memory API is not implemented yet.

## Environment

Install and sync dependencies with `uv`:

```bash
uv sync
```

Run a quick import check:

```bash
uv run python -c "import dspy; import dspy_memory; print('ok')"
```

Register the local Jupyter kernel:

```bash
uv run python -m ipykernel install --user --name dspy-memory --display-name "Python (dspy-memory)"
```

## Planned Direction

The intended authoring style is:

```python
import dspy_memory as dm

schema = dm.schema("topic")
memory = dm.Claude(schema)
```

Future work will add the actual schema layer, workflow layer, semantic
operators, storage backends, and example policies. Until then, this repository
should not claim runtime memory behavior.
