# Instruction Placeholders

How a `{...}` in an instruction is recognised, and what it means.

## 0. Dispatch: by operator, not by text

| Operator                                                                                       | Rule set                            |
| ---------------------------------------------------------------------------------------------- | ----------------------------------- |
| `sem_join`, `sem_join(k=...)`                                                              | join: two inputs are in scope       |
| `sem_filter` / `sem_map` / `sem_flat_map` / `sem_agg` / `sem_topk` / `sem_groupby` | single input: one input is in scope |

The same text means different things in the two contexts, and the operator alone
decides which. Only a join has two inputs for a side to point at.

| Instruction                              | Context      | Reads as                                                 |
| ---------------------------------------- | ------------ | -------------------------------------------------------- |
| `Is {fact:left} outdated?`             | single input | the column named `fact:left`                           |
| `Does {fact:left} match {fact:right}?` | join         | the left row's `fact` against the right row's `fact` |

## 1. Splitting

| Written                       | Result                                                    |
| ----------------------------- | --------------------------------------------------------- |
| `{{X}}`                     | literal `{X}`                                           |
| `{{{{X}}}}`                 | literal `{{X}}`                                         |
| `{X}`                       | a reference naming `X`                                  |
| `{{{X}}}`                   | literal `{`, then the reference, then literal `}`     |
| `{"a": {X}}`                | the outer braces are literal; only `{X}` is a reference |
| `{` that opens no reference | literal `{`                                             |
| `}` that closes nothing     | literal `}`                                             |

Read left to right:

1. `{{` and `}}` are literal braces.
2. A `{` whose `}` comes before any other `{` opens a reference; otherwise it is a
   literal `{`.
3. A `}` that closes nothing is literal.

`X` is the whole text between the braces. The splitter reads it as a name only when
it has the variable shape below: `a.b` and `my col` are not names, so `{a.b}` and
`{my col}` are refused rather than passed through as text.

Names you declare (log columns, `output_cols`, relation/view names, aliases) follow
the variable rule `[A-Za-z_][A-Za-z0-9_]*` (not starting with a digit), so every
declared name can be written in a reference. Names a join generates add `:` segments
under the same rule, such as `fact:left` or `fact:left:left`, so the splitter reads
them and the schema decides whether they exist.

Literal braces are doubled, so an instruction quoting JSON writes `{{"k": 1}}`.

## 2. Single-input rules

```
X = the text between the braces;  a suffix means nothing here
I = the input columns the operator supplies a value for
O = the output columns the operator expects the model to produce

① X ∈ I          → the reference names that value
② X ∈ O, X ∉ I   → the reference names a column to fill; the text stays as written
③ otherwise      → error, naming the reference
```

`I` and `O` are the operator's vocabulary: `I` is what reaches the model as data
(every column, or the ones given in `input_cols`), and `O` is what the operator
asks the model to produce. Both are columns; only `I` has a value yet. A name in
both is an input, because a value exists. `O` exists only for `sem_map`,
`sem_flat_map` and `sem_agg`; `sem_filter`, `sem_groupby` and `sem_topk` have no
output columns.

A reference to a column the table has but `I` does not is an error, because the
model has no value for it: put that column in `input_cols` first.

These operators share field resolution, not prompt layout: `sem_agg` sees a group,
`sem_groupby` sees the records it compares, and `sem_topk` sees candidates.

An instruction may name no column at all: the two sets are consulted only when a
reference appears. A reference names a value, it does not decide what the model
receives: every `I` column is supplied either way.

A column no reference names is supplied after the instruction, one line per column
as `name: value`, in the order the operator lists them. A column is supplied once:
inline where a reference names it, and in those trailing lines otherwise.

## 3. Join rules

```
X        = the text between the braces
L, R     = the two inputs
suffixes = {left → L, right → R, alias(L) → L, alias(R) → R}
           alias(L) is the alias on the expression passed as this join's left input;
           only its outermost alias counts, and an alias from an earlier join is not
           inherited
           a suffix that maps to two inputs is an error, checked when the join is built
A:S      = X split at its last colon;  S = null when X has none

check in order, first match wins

① X = A:S, S ∈ suffixes, A ∈ columns(S)      → sided reference (S, A)
② X ∈ columns(L) ∪ columns(R), exactly one   → whole-name reference
③ X ∈ columns(L) ∩ columns(R)                → error, add a suffix
④ otherwise                                  → error

a join instruction may have no reference     → the whole rows are compared
```

If the references resolve to only one side, the other side has no compared column,
so the join compares the whole rows. This is allowed; it is not an error.

The examples use `facts` (`id, fact, valid_at`) and `claims` (`id, fact`). `id` and
`fact` are on both sides, so the join has to tell them apart; `valid_at` is on one.

```python
facts.sem_join(claims, instruction="Is {fact:left} the same as {fact:right}?")
# both sides are named, so each `fact` is found
# output: id:left, fact:left, valid_at, id:right, fact:right
```

```python
facts.sem_join(claims, instruction="Is {valid_at} older than {fact:right}?")
# `valid_at` is on one side only, so its side follows from the column
# output: id:left, fact:left, valid_at, id:right, fact:right
```

```python
facts.alias("earlier").sem_join(claims, instruction="Is {fact:earlier} older than {fact:right}?")
# `earlier` names the left input, `right` still names the right one
# output: id:left, fact:left, valid_at, id:right, fact:right
```

```python
facts.alias("left").sem_join(claims, instruction="Is {fact:left} outdated?")
# the alias repeats the side's own suffix, so nothing is ambiguous
# output: id:left, fact:left, valid_at, id:right, fact:right
```

```python
facts.alias("right").sem_join(claims, instruction="Is {fact:right} outdated?")
# `right` would name both inputs, so the side cannot be inferred: error
# output: none
```

```python
facts.alias("x").sem_join(claims.alias("x"), instruction="Is {fact:x} outdated?")
# `x` would name both inputs, so the side cannot be inferred: error
# output: none
```

An alias is a variable name: ASCII letters, digits and `_`, not starting with a
digit. `earlier` is valid; `fact:left` is not. Since an alias never contains `:`,
the alias suffixes above are unambiguous. `left` and `right` always name their own
side.

- An alias renames nothing: `alias("a").alias("b")` keeps `fact`, and only the
  outermost alias `b` is consulted.
- The join writes the suffix: `left`/`right` in a semantic join, the alias or
  `left`/`right` in a deterministic one.
- Successive joins stack them into `fact:left:left`, which is why ① splits at the
  last colon.
- A reference is resolved against the two input schemas. A suffix the join will
  publish is not a side here.

```python
# a deterministic join is what puts `earlier` into a column name
joined = facts.alias("earlier").join(claims, on=["id"])   # fact:earlier, fact:right

joined.sem_join(claims, instruction="Is {fact:earlier} still true?")
# {fact:earlier} is a column name: `earlier` is no side here
```

## 4. Examples: one instruction

A single-input instruction describes whatever its operator feeds it: one row, a
group of rows, or a set of candidates. A reference always names a column of that
input.

| Instruction                             | Reads as                                                                                         |
| --------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `Is {note} positive?`                 | the `note` column                                                                              |
| `Is {fact:earlier} true?`             | the column named `fact:earlier`; `earlier` is part of the name, not a side                   |
| `Is {fact:left:left} true?`            | the column named `fact:left:left`, generated by stacked joins                                  |
| `Extract, filling {name} from {text}` | `{text}` is a value; `name` is declared as an output and is not also an input, so `{name}` stays as text |
| `Is {nott} true?`                     | neither an input nor an output column, so it is refused                                          |
| `Answer {"k": 1} when unsure`         | likewise refused; a literal brace is written `{{"k": 1}}`                                      |

A join instruction describes two rows, so a suffix can also name a side.

| Instruction                                                        | Reads as                                                 |
| ------------------------------------------------------------------ | -------------------------------------------------------- |
| `Does {fact:left} contradict {fact:right}?`                      | the left row's `fact` against the right row's `fact` |
| `Is {note} the same on both rows?`, when both rows have `note` | error: qualify it as `{note:left}` or `{note:right}` |
| `Are these two rows about the same thing?`                       | no reference, so the whole rows are compared             |

### An instruction with no reference

No reference is required. A query-only operator has no instruction text to resolve,
and a plain-language instruction is still given every selected input column:

```python
topics.sem_topk(UserQuery(), 5)          # no instruction text, nothing to resolve
items.sem_filter(instruction="Keep rows that look like a complaint")
# no reference, so the operator supplies every input column as data
```

## 5. Examples: across operators

A reference can outlive the operator that produced the name, so some cases only
make sense next to the call that made it.

### A name made by a deterministic join

```python
joined = facts.alias("earlier").join(claims.alias("later"), on=["id"])
# columns: id, fact:earlier, valid_at, fact:later

joined.sem_filter(instruction="Does {fact:later} contradict {fact:earlier}?")
# `later` is part of those column names, not a side: this is a single-input instruction
```

### The same name inside a join instruction

```python
joined.sem_join(claims, instruction="Does {fact:later} contradict {fact:right}?")
# {fact:later}  -> the column `fact:later`, on the side that has it
# {fact:right}  -> the right input's `fact`
```

### An alias on a join input

```python
facts.alias("earlier").sem_join(
    claims.alias("later"),
    instruction="Is {fact:earlier} the same as {fact:later}?",
)
# {fact:earlier} -> the left input's `fact`; the alias names the side
# the columns this join publishes are still suffixed :left and :right
```

### A column whose name ends in a side word

```python
# the left input has both `fact` and `fact:left`
joined.sem_join(claims, instruction="Is {fact:left:left} still true?")
# {fact:left} would mean the left input's `fact`;
# the extra segment names the column `fact:left`
```

## 6. Output columns

Which columns a join suffixes depends on how it matches. A semantic join suffixes
every column that appears on both sides; a deterministic join with `on=[keys]`
suffixes the overlapping columns that are not keys, and one with a predicate
suffixes every column. A key keeps its plain name, and the right side's copy of it
is not repeated.

| Situation | Published name |
| --- | --- |
| `fact` is on both sides | `fact:left` and `fact:right`, unless it is a join key |
| `valid_at` is on one side only | `valid_at`; a predicate join suffixes it as well |
| a suffixed name still matches another published name | error; one column is never dropped |

The suffix names the side the value came from, not an alias the input carried. An
alias changes how the instruction names a side; it does not change the columns the
join publishes.

A later join can suffix an already suffixed name again, so `note:left:left` can
appear. §3 says how to reference it.

## 7. Known limits

| Limit                                                            | Reason                                                             |
| ---------------------------------------------------------------- | ------------------------------------------------------------------ |
| a column whose name contains `{` or `}` cannot be referenced | the brace would end the placeholder                                |
| a brace pair that names no column is refused; `{0}`, `\frac{a}{b}`, `{"k": 1}` need doubling when those names are not columns | a brace pair is a reference; write `{{...}}` for literal braces |
| a name present on both join sides must be qualified              | two columns would otherwise answer to one reference                |
| `{%` followed by `endraw` cannot appear in an instruction             | it would end the renderer's literal block, which is how text reaches the model unchanged |