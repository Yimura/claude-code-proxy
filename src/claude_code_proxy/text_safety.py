"""Context-specific encoding for untrusted text boundaries."""

from collections.abc import Callable
import unicodedata


TELEMETRY_MODEL_MAX_LENGTH = 256
TELEMETRY_ATTRIBUTE_MAX_LENGTH = 64
TELEMETRY_BLANK_TEXT = "<blank>"


def scalar_text(value: str) -> str:
    """Return Unicode scalar text, escaping each unpaired surrogate."""
    return "".join(_scalar_atom(character) for character in value)


def _scalar_atom(character: str) -> str:
    codepoint = ord(character)
    if 0xD800 <= codepoint <= 0xDFFF:
        return unicode_escape_atom(character)
    return character


def escaped_text_atom(character: str) -> str:
    """Encode one nonprintable atom without changing printable text."""
    codepoint = ord(character)
    if codepoint < 0x20 or 0x7F <= codepoint <= 0x9F:
        return f"\\x{codepoint:02x}"
    if character.isprintable():
        return character
    return unicode_escape_atom(character)


def unicode_escape_atom(character: str) -> str:
    """Encode one code point as a printable Unicode escape."""
    codepoint = ord(character)
    if codepoint <= 0xFFFF:
        return f"\\u{codepoint:04x}"
    return f"\\U{codepoint:08x}"


def log_text(value: str) -> str:
    """Encode untrusted text so one value cannot forge log structure."""
    return "".join(_log_atom(character) for character in value)


def retained_telemetry_text(value: str, *, max_length: int) -> str:
    """Return bounded printable telemetry text without splitting escapes."""
    if type(value) is not str:
        raise TypeError("telemetry text must be a string")
    retained = TELEMETRY_BLANK_TEXT if not value.strip() else value
    return _bounded_log_value(
        retained,
        max_length=max_length,
        encode_atom=escaped_text_atom,
        exact_suffix=True,
    )


def bounded_log_text(value: str, *, max_length: int) -> str:
    """Encode untrusted log text and bound its rendered length."""
    return _bounded_log_value(value, max_length=max_length, encode_atom=log_text)


def bounded_log_token(value: str, *, max_length: int) -> str:
    """Encode one unquoted structured-log token and bound its rendered length."""
    return _bounded_log_value(
        value, max_length=max_length, encode_atom=_log_token_atom
    )


def _bounded_log_value(
    value: str,
    *,
    max_length: int,
    encode_atom: Callable[[str], str],
    exact_suffix: bool = False,
) -> str:
    if max_length < 3:
        raise ValueError("max_length must be at least 3")

    atoms: list[str] = []
    rendered_length = 0
    for character in value:
        atom = encode_atom(character)
        if rendered_length + len(atom) > max_length:
            return _truncated_log_text(
                atoms,
                rendered_length,
                max_length,
                exact_suffix=exact_suffix,
            )
        atoms.append(atom)
        rendered_length += len(atom)
    return "".join(atoms)


def _truncated_log_text(
    atoms: list[str],
    rendered_length: int,
    max_length: int,
    *,
    exact_suffix: bool,
) -> str:
    prefix_limit = max_length - 3
    while rendered_length > prefix_limit:
        rendered_length -= len(atoms.pop())
    suffix = "..." if exact_suffix else "." * (
        max_length - rendered_length
    )
    return "".join(atoms) + suffix


def _log_atom(character: str) -> str:
    codepoint = ord(character)
    if codepoint < 0x20 or 0x7F <= codepoint <= 0x9F:
        return f"\\x{codepoint:02x}"
    if (
        0xD800 <= codepoint <= 0xDFFF
        or codepoint in (0x2028, 0x2029)
        or unicodedata.category(character) == "Cf"
    ):
        return unicode_escape_atom(character)
    return character


def _log_token_atom(character: str) -> str:
    log_atom = _log_atom(character)
    if log_atom != character:
        return log_atom
    if character.isspace() or character in "=\\'\"":
        codepoint = ord(character)
        if codepoint <= 0x7F:
            return f"\\x{codepoint:02x}"
        return unicode_escape_atom(character)
    return character
