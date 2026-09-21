"""Filtering helpers for public session inventory snapshots."""

from typing import Protocol, TypeVar

from .performance_filters import AmbiguousSessionId


class PublicSnapshot(Protocol):
    id: str


SnapshotT = TypeVar("SnapshotT", bound=PublicSnapshot)


def filter_snapshots(
    snapshots: list[SnapshotT],
    filters: dict[str, tuple[str, ...]],
    exact_ids: set[str] | None,
) -> list[SnapshotT]:
    selected_ids = _resolve_id_filters(snapshots, filters.get("id"))
    matches = _filter_snapshot_ids(snapshots, selected_ids, exact_ids)
    for key, values in filters.items():
        if key not in {"id", "session_id"}:
            matches = _filter_snapshot_field(matches, key, values)
    return matches


def _filter_snapshot_ids(
    snapshots: list[SnapshotT],
    selected_ids: set[str] | None,
    exact_ids: set[str] | None,
) -> list[SnapshotT]:
    matches = snapshots
    if selected_ids is not None:
        matches = [item for item in matches if item.id in selected_ids]
    if exact_ids is not None:
        matches = [item for item in matches if item.id in exact_ids]
    return matches


def _filter_snapshot_field(
    snapshots: list[SnapshotT],
    key: str,
    values: tuple[str, ...],
) -> list[SnapshotT]:
    accepted = {value.casefold() for value in values}
    return [
        item
        for item in snapshots
        if str(getattr(item, key)).casefold() in accepted
    ]


def _resolve_id_filters(
    snapshots: list[SnapshotT],
    prefixes: tuple[str, ...] | None,
) -> set[str] | None:
    if prefixes is None:
        return None

    selected: set[str] = set()
    for prefix in prefixes:
        folded = prefix.casefold()
        matches = [
            item.id
            for item in snapshots
            if item.id.casefold().startswith(folded)
        ]
        if len(matches) > 1:
            raise AmbiguousSessionId(
                f"session ID prefix {prefix!r} is ambiguous"
            )
        selected.update(matches)
    return selected
