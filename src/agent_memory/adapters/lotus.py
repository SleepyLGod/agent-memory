"""LOTUS execution adapter shell."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_memory.logical import ColumnSpec, QueryExpr

DEFAULT_LOTUS_MODEL = "deepseek/deepseek-v4-pro"


@dataclass
class LotusAdapter:
    """Execution adapter shell for future LOTUS-backed semantic operators.

    The current implementation supports row-local sem_filter/sem_map
    maintenance plus sem_topk query execution.
    """

    model: str = DEFAULT_LOTUS_MODEL
    _configured: bool = field(default=False, init=False, repr=False)

    def execute(self, query: QueryExpr, inputs: Mapping[str, Any]) -> Any:
        """Execute a logical query expression through LOTUS."""

        match query.op:
            case "log":
                return self._input_frame(inputs, "log")
            case "materialized_view":
                return self._input_frame(inputs, str(query.params["name"]))
            case "select":
                return self._execute_select(query, inputs)
            case "sem_filter":
                return self._execute_sem_filter(query, inputs)
            case "sem_map":
                return self._execute_sem_map(query, inputs)
            case "sem_topk":
                return self._execute_sem_topk(query, inputs)
            case _:
                raise NotImplementedError(
                    f"LOTUS adapter does not support QueryExpr op {query.op!r}."
                )

    def _input_frame(self, inputs: Mapping[str, Any], name: str) -> Any:
        """Return an input DataFrame by runtime state key."""

        try:
            return inputs[name]
        except KeyError as error:
            raise KeyError(f"Missing adapter input {name!r}") from error

    def _execute_select(self, query: QueryExpr, inputs: Mapping[str, Any]) -> Any:
        """Execute deterministic column projection."""

        source = self.execute(query.inputs[0], inputs)
        return source.loc[:, list(query.params["columns"])].copy()

    def _execute_sem_filter(self, query: QueryExpr, inputs: Mapping[str, Any]) -> Any:
        """Execute LOTUS semantic filtering."""

        self._configure_lotus()
        source = self.execute(query.inputs[0], inputs)
        return source.sem_filter(query.params["instruction"])

    def _execute_sem_topk(self, query: QueryExpr, inputs: Mapping[str, Any]) -> Any:
        """Execute LOTUS semantic top-k retrieval."""

        self._configure_lotus()
        source = self.execute(query.inputs[0], inputs)
        instruction = self._topk_instruction(
            source,
            str(query.params["instruction"]),
        )
        return source.sem_topk(
            instruction,
            K=query.params["k"],
            method="quick",
        )

    def _topk_instruction(self, source: Any, instruction: str) -> str:
        """Convert plain user queries into LOTUS column-aware expressions."""

        if "{" in instruction and "}" in instruction:
            return instruction

        columns = [str(column) for column in getattr(source, "columns", ())]
        if not columns:
            raise ValueError("sem_topk requires at least one input column")

        row_reference = ", ".join(f"{{{column}}}" for column in columns)
        return f"{row_reference} is relevant to: {instruction}"

    def _execute_sem_map(self, query: QueryExpr, inputs: Mapping[str, Any]) -> Any:
        """Execute single-output sem_map through LOTUS native string output."""

        output_col = self._single_output_column(query)
        self._configure_lotus()
        source = self.execute(query.inputs[0], inputs)
        map_column = self._temporary_map_column(source)
        mapped = source.sem_map(str(query.params["instruction"]), suffix=map_column)
        return self._apply_sem_map_output(source, mapped, map_column, output_col)

    def _apply_sem_map_output(
        self,
        source: Any,
        mapped: Any,
        map_column: str,
        output_col: ColumnSpec,
    ) -> Any:
        """Copy LOTUS native sem_map string output into the requested column."""

        result = source.copy()
        result[output_col.name] = mapped[map_column]
        return result

    def _temporary_map_column(self, source: Any) -> str:
        """Return a LOTUS sem_map output column that will not overwrite input."""

        existing = set(str(column) for column in getattr(source, "columns", ()))
        base = "_agent_memory_map"
        candidate = base
        suffix = 1
        while candidate in existing:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    def _output_columns(self, query: QueryExpr) -> tuple[ColumnSpec, ...]:
        """Return required output columns for sem_map."""

        output_cols = query.params.get("output_cols")
        if not output_cols:
            raise ValueError("sem_map requires output_cols")
        return tuple(output_cols)

    def _single_output_column(self, query: QueryExpr) -> ColumnSpec:
        """Return the one output column supported by LOTUS native sem_map."""

        output_cols = self._output_columns(query)
        if len(output_cols) != 1:
            raise NotImplementedError(
                "LotusAdapter currently supports one output column per sem_map; "
                "chain sem_map calls or implement structured map lowering."
            )
        return output_cols[0]

    def _configure_lotus(self) -> None:
        """Configure LOTUS with the selected LiteLLM-compatible model."""

        if self._configured:
            return

        import lotus
        from lotus.models import LM

        lotus.settings.configure(lm=LM(model=self.model))
        self._configured = True
