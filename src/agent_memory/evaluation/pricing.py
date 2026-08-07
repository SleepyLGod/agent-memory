"""Reproducible provider pricing used by benchmark metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class PricingSnapshot:
    """Immutable DeepSeek pricing used for reproducible cost estimates."""

    effective_date: str
    source_url: str
    cache_hit_input_per_million_usd: Decimal
    cache_miss_input_per_million_usd: Decimal
    output_per_million_usd: Decimal

    @classmethod
    def deepseek_2026_07_17(cls) -> "PricingSnapshot":
        """Return the pricing snapshot shared by all benchmark systems."""

        return cls(
            effective_date="2026-07-17",
            source_url="https://api-docs.deepseek.com/quick_start/pricing",
            cache_hit_input_per_million_usd=Decimal("0.0028"),
            cache_miss_input_per_million_usd=Decimal("0.14"),
            output_per_million_usd=Decimal("0.28"),
        )

    def estimate_cost_usd(
        self,
        *,
        cache_hit_input_tokens: int | None,
        cache_miss_input_tokens: int | None,
        output_tokens: int | None,
    ) -> Decimal | None:
        """Estimate cost only when every billable counter is available."""

        if (
            cache_hit_input_tokens is None
            or cache_miss_input_tokens is None
            or output_tokens is None
        ):
            return None
        million = Decimal(1_000_000)
        return (
            Decimal(cache_hit_input_tokens)
            * self.cache_hit_input_per_million_usd
            + Decimal(cache_miss_input_tokens)
            * self.cache_miss_input_per_million_usd
            + Decimal(output_tokens) * self.output_per_million_usd
        ) / million

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe pricing manifest."""

        return {
            key: str(value) if isinstance(value, Decimal) else value
            for key, value in asdict(self).items()
        }


__all__ = ["PricingSnapshot"]
