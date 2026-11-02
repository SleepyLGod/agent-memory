"""The record keys this adapter injects, and the check that keeps them out of a caller's frame."""

from __future__ import annotations

from collections.abc import Iterable

# Required by: the record contract (`docetl.operations.base::BaseOperation.execute`) and the runtime's
#   row-local nodes, which file results by occurrence
#   (`runtime/executor.py::PolicyExecutor._execute_semantic_row`).
INDEX_KEY = "__index__"

# Required by: `sem_filter.py::build_filter_config` and `sem_flat_map.py::build_flat_map_config`. In the
#   prompt these are the fields the model fills: `keep_row` is the boolean verdict, `output_rows` the list
#   of rows that input row produces.
KEEP_KEY = "keep_row"
ROWS_KEY = "output_rows"

# Required by: `reject_reserved`, which `to_records` runs over every frame entering an operator. These
#   names appear in the model's tool schema, so a policy column may not shadow one of them.
RESERVED_NAMES = frozenset({INDEX_KEY, KEEP_KEY, ROWS_KEY})


def reject_reserved(columns: Iterable[str]) -> None:
    """Raise if any of `columns` is a name this adapter injects."""

    clashes = [str(column) for column in columns if str(column) in RESERVED_NAMES]
    if clashes:
        raise ValueError(f"frame uses adapter-reserved column names: {clashes}")
