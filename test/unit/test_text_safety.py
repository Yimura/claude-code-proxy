import pytest

import claude_code_proxy.text_safety as text_safety


def test_bounded_log_text_returns_short_encoded_text_unchanged():
    assert (
        text_safety.bounded_log_text("safe\ntext", max_length=20)
        == "safe\\x0atext"
    )


@pytest.mark.parametrize(
    ("value", "escape_prefix"),
    [
        ("alpha\nbravo", "\\x"),
        ("alpha bravo", "\\u"),
        ("alpha\U000E0001bravo", "\\U"),
    ],
)
def test_bounded_log_text_truncates_at_complete_escape_atoms(value, escape_prefix):
    result = text_safety.bounded_log_text(value, max_length=10)

    assert len(result) == 10
    assert result.endswith("...")
    assert not any(character in result for character in "\n \U000E0001")
    assert escape_prefix not in result


def test_bounded_log_text_stops_encoding_after_truncation_is_known(monkeypatch):
    calls = []

    def record_encoded_character(character):
        calls.append(character)
        return character

    monkeypatch.setattr(text_safety, "log_text", record_encoded_character)

    result = text_safety.bounded_log_text("a" * 1_000_000, max_length=10)

    assert result == "a" * 7 + "..."
    assert calls == ["a"] * 11


@pytest.mark.parametrize("max_length", [0, 2])
def test_bounded_log_text_rejects_lengths_too_short_for_suffix(max_length):
    with pytest.raises(ValueError):
        text_safety.bounded_log_text("value", max_length=max_length)


def test_bounded_log_token_escapes_whitespace_and_field_delimiters():
    value = "a b=c\\d'e\"f\u00a0g\n"

    result = text_safety.bounded_log_token(value, max_length=100)

    assert result == (
        "a\\x20b\\x3dc\\x5cd\\x27e\\x22f"
        "\\u00a0g\\x0a"
    )
    assert len(result.split()) == 1
    assert "=" not in result
    assert "'" not in result
    assert '"' not in result


def test_bounded_log_token_preserves_complete_atoms_at_exact_bound():
    result = text_safety.bounded_log_token(
        "prefix status=200\u00a0" + "x" * 300,
        max_length=128,
    )

    assert len(result) == 128
    assert result.endswith("...")
    assert "\\x20status\\x3d200\\u00a0" in result
    assert not result.endswith(("\\", "\\x", "\\u"))
