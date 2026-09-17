# Codex Tool Schema Strictness

## Problem

Claude Code tool schemas use omission to represent optional fields. Codex function tools currently forward those schemas without an explicit `strict` value. In observed `gpt-5.6-sol` calls, every declared `Monitor` property was populated, so command-source calls also contained a fabricated `ws` object and failed Claude Code's exactly-one-of runtime validator.

Current boundary:

```text
Claude Code input_schema
        ↓ unchanged
Codex Responses function tool without strict
        ↓ model populates optional placeholders
Claude Code runtime refinement rejects command + ws
```

## Constraints

- Preserve each inbound `input_schema` unchanged.
- Preserve omission semantics for optional properties.
- Avoid tool-name-specific cleanup.
- Avoid strict-schema normalization to required nullable properties.
- Preserve existing tool choice, reasoning replay, and non-Codex behavior.
- Match official Codex dynamic/MCP tool behavior.

## Design

`src/claude_code_proxy/providers/codex/translation.py::build_request()` will emit every translated function tool with explicit non-strict behavior:

```python
{
    "type": "function",
    "name": tool.name,
    "description": tool.description,
    "strict": False,
    "parameters": tool.input_schema,
}
```

No schema copy or transformation is required. The existing domain model remains unchanged. Provider scoping is inherent because only the Codex translator builds this payload.

## Data flow

```text
ToolDefinition.input_schema
        ↓
build_request()
        ├── parameters: same schema object/content
        └── strict: false
        ↓
Codex best-effort function calling
        ↓
Tool arguments preserve omission-based optional branches
        ↓
CodexEventTranslator forwards returned JSON unchanged
```

## Error behavior

This change does not weaken Claude Code's runtime validation. Invalid model output still reaches normal tool validation and fails visibly. It only prevents strict-mode assumptions from being inferred for schemas that do not satisfy strict-mode requirements.

No null stripping, placeholder cleanup, or retry behavior is added.

## Verification

Deterministic unit coverage will prove:

1. Every translated Codex function tool includes `strict: false`.
2. Original parameter schema remains unchanged.
3. Command-source `Monitor` arguments preserve `command` and omit `ws` when upstream emits that shape.
4. WebSocket-source `Monitor` arguments preserve `ws` and omit `command` when upstream emits that shape.
5. Existing tool choice and encrypted reasoning replay tests remain green.
6. Full offline test suite passes.

No live model evaluation is required for this change because user is unavailable to perform one. Issue evidence already provides 16/16 failing proxied calls, and deterministic tests cover both proxy boundaries changed or relied upon by the fix.

## Alternatives rejected

### Strict-normalize every schema

Strict mode requires all properties in `required`, `additionalProperties: false`, and nullable unions for optional values. Converting arbitrary Claude Code schemas would alter omission semantics and require response-side null cleanup before runtime validation.

### Remove conflicting fields from Monitor calls

A Monitor-specific sanitizer would patch one symptom while leaving every omission-sensitive tool vulnerable. It would also hardcode Claude Code tool behavior inside generic response translation.

### Leave `strict` unspecified

Current behavior is deterministic in observed Codex sessions and diverges from official Codex dynamic/MCP tool translation.

## References

- OpenAI function calling strict mode: https://developers.openai.com/api/docs/guides/function-calling#strict-mode
- Official Codex Responses tool translation: https://github.com/openai/codex/blob/7abf2a3b5cbe08ca875d677dcd027528f9556152/codex-rs/tools/src/responses_api.rs#L157-L165
- Tracker: https://github.com/Yimura/claude-code-proxy/issues/44
