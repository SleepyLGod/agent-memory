# Semantic maintenance: branch review map

This branch collects the incremental implementation work on semantic maintenance.
The dated experiment/design notes describe individual iterations, not one uniform
benchmark condition. This page is the entry point for reviewing the final code.

## Scope and semantic boundaries

| Area | Implementation | Contract |
| --- | --- | --- |
| Physical planning | `planner/physical.py` | Explicit registered strategies; disabled by default. No cost-based search. |
| Structured execution | `adapters/lotus/fusion.py`, `site_batching.py` | Target-state fusion and node-scoped packing change prompts, not a proof of LLM output equivalence. |
| Decision reuse | `predicate_reuse.py`, `identity_reuse.py` | Reuse semantic decisions under the declared dependency contract; propagate current row metadata. Not provider caching. |
| Runtime state | `runtime/executor.py`, `row_outputs.py` | Incremental relational joins, indexed candidate lookup, shared empty outputs and identity/difference calculation preserve bag identities. |
| Scheduling | `scoped_lm.py` and independent extraction | Independent extraction has isolated execution context; dependent maintenance remains ordered. |
| Evaluation | `evaluation/` | Cache-aware usage attribution, execution provenance, answer validation and explicit cache configuration. |

Two Zep variants are **logical changes**, not invisible physical optimizations:

- `ZepFactSummaryMemory` maintains entity summaries from associated facts.
- `ZepRepresentativeMemory` retains a deterministically selected representative
  fact instead of synthesizing a new canonical text for duplicate groups.

These variants must be compared as named configurations. They are not claims that
the original Zep view or Native Graphiti is reproduced exactly. Singleton fact
passthrough and summary truncation also have explicit quality tradeoffs.

## Review order

1. Read `planner/physical.py` for registration, eligibility and fingerprints.
2. Read the adapter execution paths for task identity, parsing and reconstruction.
3. Review runtime insertion/retraction, predicate reuse and snapshot tests together.
4. Review the two Zep view variants separately from physical execution changes.
5. Review benchmark provenance and accounting before interpreting experiment data.

The original first-iteration design is in
[semantic-physical-fusion.zh.md](semantic-physical-fusion.zh.md). Later boundaries
are described in [zep-combined-physical.zh.md](zep-combined-physical.zh.md),
[zep-fact-summary.zh.md](zep-fact-summary.zh.md), and
[zep-native-alignment.zh.md](zep-native-alignment.zh.md).
Runtime changes are documented in
[semantic-output-cache.zh.md](../optimization/semantic-output-cache.zh.md) and
[zep-replacement-maintenance-fix.zh.md](../optimization/zep-replacement-maintenance-fix.zh.md).

## Validation and limits

Run the offline suite with the repository source on `PYTHONPATH`:

```sh
PYTHONPATH=src python -m pytest -q
ruff check src tests tools
pyright
git diff --check
```

The configured Pyright target set is narrower than all source files. Skipped
integration tests and deterministic fake providers do not establish real model
quality, latency, server recovery or equivalence to Native. This PR review does
not start a paid experiment or alter historical artifacts.

Execution fingerprints reject incompatible snapshots. This is not an exactly-once
provider billing guarantee. LOTUS caches are process-local and cold after restart.
Fusion retains its explicit bounded repair behavior (at least one parse retry),
even when the generic structured retry setting is zero; experiments must account
for actual calls rather than infer them from that generic setting alone.

## Review correction

Execution provenance now records the selected fusion strategy's actual version,
even without site-level options. Site options alone no longer claim that fusion
ran. Ten regression combinations cover disabled and all four registered strategies,
both with and without site options. This corrects metadata, not historical results.
