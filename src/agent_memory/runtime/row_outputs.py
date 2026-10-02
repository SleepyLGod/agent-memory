"""Private, transaction-local indexing of semantic occurrence outputs."""

from collections.abc import Hashable, Iterator, Mapping, MutableMapping

import pandas as pd


class RowOutputCache(MutableMapping[Hashable, pd.DataFrame]):
    """Retain empty membership cheaply and enumerate only nonempty outputs.

    Stored frames are read-only by convention, just like executor node state.
    Updates replace frames; forks share frames but never mutable bookkeeping.
    Snapshots serialize the mapping, not this derived index.
    """

    def __init__(self, frames: Mapping[Hashable, pd.DataFrame] | None = None) -> None:
        self._frames: dict[Hashable, pd.DataFrame] = {}
        self._positions: dict[Hashable, int] = {}
        self._nonempty: dict[Hashable, pd.DataFrame] = {}
        self._next_position = 0
        self._empty_templates: list[pd.DataFrame] = []
        if frames is not None:
            for key, frame in frames.items():
                self[key] = frame

    def __getitem__(self, key: Hashable) -> pd.DataFrame:
        return self._frames[key]

    def __setitem__(self, key: Hashable, frame: pd.DataFrame) -> None:
        empty = frame.empty
        if empty:
            frame = self._intern_empty(frame)
        if key not in self._frames:
            self._positions[key] = self._next_position
            self._next_position += 1
        self._frames[key] = frame
        if empty:
            self._nonempty.pop(key, None)
        else:
            self._nonempty[key] = frame

    def __delitem__(self, key: Hashable) -> None:
        del self._frames[key]
        del self._positions[key]
        self._nonempty.pop(key, None)

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._frames)

    def __len__(self) -> int:
        return len(self._frames)

    def copy(self) -> "RowOutputCache":
        """Fork bookkeeping without copying shared, unmodified output frames."""
        result = RowOutputCache()
        result._frames = self._frames.copy()
        result._positions = self._positions.copy()
        result._nonempty = self._nonempty.copy()
        result._next_position = self._next_position
        result._empty_templates = self._empty_templates.copy()
        return result

    def empty_like(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Get a read-only template once before recording a batch of empties."""
        return self._intern_empty(frame.iloc[:0])

    def nonempty_frames(self) -> list[pd.DataFrame]:
        """Return outputs in mapping insertion order, without visiting empties."""
        return [self._nonempty[key] for key in sorted(self._nonempty, key=self._positions.__getitem__)]

    def _intern_empty(self, frame: pd.DataFrame) -> pd.DataFrame:
        for template in self._empty_templates:
            if frame is template:
                return template
        # Arbitrary attrs may contain mutable/non-comparable values. Keep those
        # frames independent rather than silently dropping metadata.
        if frame.attrs:
            return frame.copy(deep=True)
        for template in self._empty_templates:
            if (
                frame.columns.identical(template.columns)
                and frame.index.identical(template.index)
                and frame.dtypes.equals(template.dtypes)
                and frame.flags.allows_duplicate_labels == template.flags.allows_duplicate_labels
            ):
                return template
        template = frame.copy(deep=True)
        self._empty_templates.append(template)
        return template
