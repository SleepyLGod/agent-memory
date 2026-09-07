"""Serializable row expressions for logical relation queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

Scalar = str | int | float | bool | None
ExprParam = Mapping[str, Any]


def is_scalar(value: object) -> bool:
    """Return whether a value is a supported expression literal."""

    return value is None or isinstance(value, (str, int, float, bool))


def literal(value: Scalar) -> "LiteralExpr":
    """Create a literal expression."""

    if not is_scalar(value):
        raise TypeError(f"Unsupported literal expression value: {type(value).__name__}")
    return LiteralExpr(value)


def least(*values: object) -> "LeastExpr":
    """Return a row-wise minimum expression that ignores null operands."""

    if len(values) < 2:
        raise ValueError("least requires at least two operands")
    return LeastExpr(operands=tuple(ensure_expr(value) for value in values))


def try_cast(value: object, *, to: str) -> "TryCastExpr":
    """Build a nullable scalar conversion expression."""

    if to != "float":
        raise ValueError("try_cast currently only supports the 'float' target")
    return TryCastExpr(value=ensure_expr(value), target=to)


def case_when(
    condition: object,
    then_value: object,
    else_value: object,
) -> "CaseWhenExpr":
    """Build a SQL-style conditional scalar expression."""

    return CaseWhenExpr(
        condition=ensure_boolean_expr(condition),
        then_value=ensure_expr(then_value),
        else_value=ensure_expr(else_value),
    )


def ensure_expr(value: object) -> "Expr":
    """Normalize supported public expression values into expression objects."""

    if isinstance(value, Expr):
        return value
    if is_scalar(value):
        return LiteralExpr(value)  # type: ignore[arg-type]
    raise TypeError(f"Expected a relational expression or scalar literal, got {type(value).__name__}")


def ensure_boolean_expr(
    value: object,
) -> "BooleanExpr | ColumnExpr | ComparisonExpr | LiteralExpr":
    """Normalize and require a boolean-valued expression."""

    expr = ensure_expr(value)
    if isinstance(expr, (BooleanExpr, ColumnExpr, ComparisonExpr)):
        return expr
    if isinstance(expr, LiteralExpr) and isinstance(expr.value, bool):
        return expr
    raise TypeError(f"Expected a boolean relational expression, got {type(expr).__name__}")


def expr_to_param(expr: object) -> ExprParam:
    """Convert an expression object into a frozen QueryExpr-compatible param."""

    return ensure_expr(expr).to_param()


def expr_from_param(value: object) -> "Expr":
    """Rebuild an expression object from a QueryExpr param."""

    if isinstance(value, Expr):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"Expression param must be a mapping, got {type(value).__name__}")
    kind = value.get("kind")
    if kind == "column":
        name = value.get("name")
        qualifier = value.get("qualifier")
        if not isinstance(name, str) or not name:
            raise TypeError("Column expression requires a non-empty string name")
        if qualifier is not None and not isinstance(qualifier, str):
            raise TypeError("Column expression qualifier must be a string or None")
        return ColumnExpr(name=name, qualifier=qualifier)
    if kind == "literal":
        literal_value = value.get("value")
        if not is_scalar(literal_value):
            raise TypeError(f"Unsupported literal expression value: {type(literal_value).__name__}")
        return LiteralExpr(literal_value)  # type: ignore[arg-type]
    if kind == "comparison":
        op = value.get("op")
        if op not in {"eq", "ne", "lt", "le", "gt", "ge"}:
            raise ValueError(f"Unsupported comparison expression op: {op!r}")
        return ComparisonExpr(
            op=op,  # type: ignore[arg-type]
            left=expr_from_param(value.get("left")),
            right=expr_from_param(value.get("right")),
        )
    if kind == "arithmetic":
        op = value.get("op")
        if op not in {"add", "subtract", "multiply", "divide"}:
            raise ValueError(f"Unsupported arithmetic expression op: {op!r}")
        return ArithmeticExpr(
            op=op,  # type: ignore[arg-type]
            left=expr_from_param(value.get("left")),
            right=expr_from_param(value.get("right")),
        )
    if kind == "try_cast":
        target = value.get("target")
        if target != "float":
            raise ValueError("try_cast currently only supports the 'float' target")
        return TryCastExpr(
            value=expr_from_param(value.get("value")),
            target=target,
        )
    if kind == "case_when":
        return CaseWhenExpr(
            condition=ensure_boolean_expr(expr_from_param(value.get("condition"))),
            then_value=expr_from_param(value.get("then")),
            else_value=expr_from_param(value.get("else")),
        )
    if kind == "boolean":
        op = value.get("op")
        operands = value.get("operands")
        if op not in {"and", "or", "not", "in", "is_null", "is_not_null"}:
            raise ValueError(f"Unsupported boolean expression op: {op!r}")
        if not isinstance(operands, Sequence) or isinstance(operands, (str, bytes)):
            raise TypeError("Boolean expression operands must be a sequence")
        return BooleanExpr(
            op=op,  # type: ignore[arg-type]
            operands=tuple(expr_from_param(operand) for operand in operands),
        )
    if kind == "array_cat":
        return ArrayCatExpr(
            left=expr_from_param(value.get("left")),
            right=expr_from_param(value.get("right")),
        )
    if kind == "least":
        operands = value.get("operands")
        if not isinstance(operands, Sequence) or isinstance(operands, (str, bytes)):
            raise TypeError("Least expression operands must be a sequence")
        if len(operands) < 2:
            raise ValueError("least requires at least two operands")
        return LeastExpr(
            operands=tuple(expr_from_param(operand) for operand in operands),
        )
    raise ValueError(f"Unsupported expression kind: {kind!r}")


@dataclass(frozen=True)
class Expr:
    """Base class for serializable relational expressions."""

    def to_param(self) -> ExprParam:
        """Return a QueryExpr parameter representation."""

        raise NotImplementedError

    def __bool__(self) -> bool:
        """Reject accidental Python truthiness for query expressions."""

        raise TypeError(
            "Relational expressions cannot be used as Python booleans. "
            "Pass them to filter/join, or compare to_param() for structural checks."
        )

    def __and__(self, other: object) -> "BooleanExpr":
        """Build a boolean AND expression."""

        return BooleanExpr(op="and", operands=(ensure_boolean_expr(self), ensure_boolean_expr(other)))

    def __or__(self, other: object) -> "BooleanExpr":
        """Build a boolean OR expression."""

        return BooleanExpr(op="or", operands=(ensure_boolean_expr(self), ensure_boolean_expr(other)))

    def __invert__(self) -> "BooleanExpr":
        """Build a boolean NOT expression."""

        return BooleanExpr(op="not", operands=(ensure_boolean_expr(self),))

    def isin(self, values: list[Scalar] | tuple[Scalar, ...]) -> "BooleanExpr":
        """Build an IN predicate."""

        if not isinstance(values, (list, tuple)):
            raise TypeError("isin values must be a list or tuple")
        return BooleanExpr(
            op="in",
            operands=(self, *(literal(value) for value in values)),
        )

    def is_null(self) -> "BooleanExpr":
        """Build an IS NULL predicate."""

        return BooleanExpr(op="is_null", operands=(self,))

    def is_not_null(self) -> "BooleanExpr":
        """Build an IS NOT NULL predicate."""

        return BooleanExpr(op="is_not_null", operands=(self,))

    def array_cat(self, other: object) -> "ArrayCatExpr":
        """Build a row-wise JSON array-state concatenation expression."""

        return ArrayCatExpr(left=self, right=ensure_expr(other))

    def __add__(self, other: object) -> "ArithmeticExpr":
        """Build an addition expression."""

        return ArithmeticExpr(op="add", left=self, right=ensure_expr(other))

    def __radd__(self, other: object) -> "ArithmeticExpr":
        """Build an addition expression with a scalar left operand."""

        return ArithmeticExpr(op="add", left=ensure_expr(other), right=self)

    def __sub__(self, other: object) -> "ArithmeticExpr":
        """Build a subtraction expression."""

        return ArithmeticExpr(op="subtract", left=self, right=ensure_expr(other))

    def __rsub__(self, other: object) -> "ArithmeticExpr":
        """Build a subtraction expression with a scalar left operand."""

        return ArithmeticExpr(op="subtract", left=ensure_expr(other), right=self)

    def __mul__(self, other: object) -> "ArithmeticExpr":
        """Build a multiplication expression."""

        return ArithmeticExpr(op="multiply", left=self, right=ensure_expr(other))

    def __rmul__(self, other: object) -> "ArithmeticExpr":
        """Build a multiplication expression with a scalar left operand."""

        return ArithmeticExpr(op="multiply", left=ensure_expr(other), right=self)

    def __truediv__(self, other: object) -> "ArithmeticExpr":
        """Build a division expression."""

        return ArithmeticExpr(op="divide", left=self, right=ensure_expr(other))

    def __rtruediv__(self, other: object) -> "ArithmeticExpr":
        """Build a division expression with a scalar left operand."""

        return ArithmeticExpr(op="divide", left=ensure_expr(other), right=self)


@dataclass(frozen=True, eq=False)
class ColumnExpr(Expr):
    """Reference to one relation column, optionally qualified by an alias."""

    __hash__ = None  # type: ignore[assignment]

    name: str
    qualifier: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Column expression name cannot be empty")
        if self.qualifier == "":
            raise ValueError("Column expression qualifier cannot be empty")

    def to_param(self) -> ExprParam:
        """Return a QueryExpr parameter representation."""

        return {"kind": "column", "name": self.name, "qualifier": self.qualifier}

    def __eq__(self, other: object) -> "ComparisonExpr":  # type: ignore[override]
        return ComparisonExpr(op="eq", left=self, right=ensure_expr(other))

    def __ne__(self, other: object) -> "ComparisonExpr":  # type: ignore[override]
        return ComparisonExpr(op="ne", left=self, right=ensure_expr(other))

    def __lt__(self, other: object) -> "ComparisonExpr":
        return ComparisonExpr(op="lt", left=self, right=ensure_expr(other))

    def __le__(self, other: object) -> "ComparisonExpr":
        return ComparisonExpr(op="le", left=self, right=ensure_expr(other))

    def __gt__(self, other: object) -> "ComparisonExpr":
        return ComparisonExpr(op="gt", left=self, right=ensure_expr(other))

    def __ge__(self, other: object) -> "ComparisonExpr":
        return ComparisonExpr(op="ge", left=self, right=ensure_expr(other))


@dataclass(frozen=True)
class LiteralExpr(Expr):
    """Scalar literal expression."""

    value: Scalar

    def to_param(self) -> ExprParam:
        """Return a QueryExpr parameter representation."""

        return {"kind": "literal", "value": self.value}


@dataclass(frozen=True)
class ComparisonExpr(Expr):
    """Binary comparison expression."""

    op: Literal["eq", "ne", "lt", "le", "gt", "ge"]
    left: Expr
    right: Expr

    def to_param(self) -> ExprParam:
        """Return a QueryExpr parameter representation."""

        return {
            "kind": "comparison",
            "op": self.op,
            "left": self.left.to_param(),
            "right": self.right.to_param(),
        }


@dataclass(frozen=True)
class ArithmeticExpr(Expr):
    """Binary numeric expression with SQL-style null propagation."""

    op: Literal["add", "subtract", "multiply", "divide"]
    left: Expr
    right: Expr

    def to_param(self) -> ExprParam:
        """Return a QueryExpr parameter representation."""

        return {
            "kind": "arithmetic",
            "op": self.op,
            "left": self.left.to_param(),
            "right": self.right.to_param(),
        }


@dataclass(frozen=True)
class TryCastExpr(Expr):
    """Nullable scalar conversion expression."""

    value: Expr
    target: Literal["float"]

    def to_param(self) -> ExprParam:
        """Return a QueryExpr parameter representation."""

        return {
            "kind": "try_cast",
            "value": self.value.to_param(),
            "target": self.target,
        }


@dataclass(frozen=True)
class CaseWhenExpr(Expr):
    """SQL-style conditional expression with an explicit else branch."""

    condition: Expr
    then_value: Expr
    else_value: Expr

    def to_param(self) -> ExprParam:
        """Return a QueryExpr parameter representation."""

        return {
            "kind": "case_when",
            "condition": self.condition.to_param(),
            "then": self.then_value.to_param(),
            "else": self.else_value.to_param(),
        }


@dataclass(frozen=True)
class BooleanExpr(Expr):
    """Boolean expression over relational expressions."""

    op: Literal["and", "or", "not", "in", "is_null", "is_not_null"]
    operands: tuple[Expr, ...]

    def to_param(self) -> ExprParam:
        """Return a QueryExpr parameter representation."""

        return {
            "kind": "boolean",
            "op": self.op,
            "operands": tuple(operand.to_param() for operand in self.operands),
        }


@dataclass(frozen=True)
class ArrayCatExpr(Expr):
    """Row-wise JSON array-state concatenation expression."""

    left: Expr
    right: Expr

    def to_param(self) -> ExprParam:
        """Return a QueryExpr parameter representation."""

        return {
            "kind": "array_cat",
            "left": self.left.to_param(),
            "right": self.right.to_param(),
        }


@dataclass(frozen=True)
class LeastExpr(Expr):
    """Row-wise minimum expression over two or more nullable operands."""

    operands: tuple[Expr, ...]

    def __post_init__(self) -> None:
        if len(self.operands) < 2:
            raise ValueError("least requires at least two operands")

    def to_param(self) -> ExprParam:
        """Return a QueryExpr parameter representation."""

        return {
            "kind": "least",
            "operands": tuple(operand.to_param() for operand in self.operands),
        }
