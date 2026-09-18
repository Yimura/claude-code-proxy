"""Context-specific encoding for untrusted text boundaries."""

import unicodedata


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
