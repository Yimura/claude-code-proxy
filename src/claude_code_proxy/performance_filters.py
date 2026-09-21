"""Stable identity resolution for filtered performance subscriptions."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Final, TypeAlias

from .event_journal import JournalEvent

FilterMap: TypeAlias = Mapping[str, Sequence[str]]
FrozenFilterMap: TypeAlias = Mapping[str, tuple[str, ...]]
FILTER_FIELDS: Final = frozenset(
    {"id", "session_id", "state", "provider", "transport", "model", "effort"}
)
_MATCH_NONE_ID = "-"


class InvalidSessionFilter(ValueError):
    pass


class AmbiguousSessionId(ValueError):
    pass


def validate_performance_filters(
    filters: FilterMap | None,
) -> dict[str, tuple[str, ...]] | None:
    if filters is None:
        return None
    if not isinstance(filters, Mapping) or not filters:
        raise InvalidSessionFilter("filters must contain at least one key")
    normalized: dict[str, tuple[str, ...]] = {}
    for key, values in filters.items():
        _validate_filter_key(key)
        normalized[key] = _validate_filter_values(key, values)
    return normalized


def _validate_filter_key(key: object) -> None:
    if not isinstance(key, str) or not key.strip() or key not in FILTER_FIELDS:
        raise InvalidSessionFilter(f"unsupported session filter {key!r}")


def _validate_filter_values(key: str, values: object) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise InvalidSessionFilter(
            f"filter {key!r} requires a sequence of values"
        )
    entries = tuple(values)
    if not entries or any(_invalid_filter_value(key, value) for value in entries):
        raise InvalidSessionFilter(
            f"filter {key!r} requires non-empty values"
        )
    return entries


def _invalid_filter_value(key: str, value: object) -> bool:
    if not isinstance(value, str) or not value:
        return True
    return key == "session_id" and not value.strip()


def prepare_performance_filters(
    filters: FilterMap | None,
    current_ids: Iterable[str],
    replay: Sequence[JournalEvent],
    public_id: Callable[[str], str],
    *,
    resolved: bool,
) -> FrozenFilterMap | None:
    if resolved:
        return freeze_performance_filters(filters)
    return resolve_performance_filters(filters, current_ids, replay, public_id)


def mutable_performance_filters(
    filters: FrozenFilterMap | None,
) -> dict[str, tuple[str, ...]] | None:
    return None if filters is None else dict(filters)


def freeze_performance_filters(
    filters: FilterMap | None,
) -> FrozenFilterMap | None:
    if filters is None:
        return None
    return MappingProxyType(
        {key: tuple(values) for key, values in filters.items()}
    )


def resolve_performance_filters(
    filters: FilterMap | None,
    current_ids: Iterable[str],
    replay: Sequence[JournalEvent],
    public_id: Callable[[str], str],
) -> FrozenFilterMap | None:
    if filters is None:
        return None
    resolved = {
        key: tuple(values)
        for key, values in filters.items()
        if key not in {"id", "session_id"}
    }
    identities = _resolve_filter_identities(
        filters,
        current_ids,
        replay,
        public_id,
    )
    if identities is not None:
        resolved["id"] = tuple(sorted(identities)) or (_MATCH_NONE_ID,)
    return MappingProxyType(resolved)


def _resolve_filter_identities(
    filters: FilterMap,
    current_ids: Iterable[str],
    replay: Sequence[JournalEvent],
    public_id: Callable[[str], str],
) -> set[str] | None:
    prefixes = filters.get("id")
    raw_session_ids = filters.get("session_id")
    prefix_ids = _resolve_prefixes(prefixes, current_ids, replay)
    exact_ids = (
        {public_id(value) for value in raw_session_ids}
        if raw_session_ids is not None
        else None
    )
    if prefix_ids is None:
        return exact_ids
    if exact_ids is None:
        return prefix_ids
    return prefix_ids & exact_ids


def _resolve_prefixes(
    prefixes: Sequence[str] | None,
    current_ids: Iterable[str],
    replay: Sequence[JournalEvent],
) -> set[str] | None:
    if prefixes is None:
        return None
    available = set(current_ids)
    available.update(event.activity.id for event in replay)
    selected: set[str] = set()
    for prefix in prefixes:
        folded = prefix.casefold()
        matches = {
            identity
            for identity in available
            if identity.casefold().startswith(folded)
        }
        if len(matches) > 1:
            raise AmbiguousSessionId(
                f"session ID prefix {prefix!r} is ambiguous"
            )
        selected.update(matches)
    return selected
