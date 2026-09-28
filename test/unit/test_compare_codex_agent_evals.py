import importlib.util
from importlib.machinery import SourceFileLoader
import json
from pathlib import Path
import subprocess
import sys

import pytest

from test.integration.codex_agent_eval_support import (
    EVAL_SCENARIO_NAMES,
    MAX_REPORT_BYTES,
    EvalReport,
    MetricValue,
    ScenarioAggregate,
)

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts" / "compare-codex-agent-evals"


def test_comparison_rejects_lower_completeness_before_token_savings():
    module = _load_script()
    baseline = _report(completeness=1.0, passed=1.0, tokens=100)
    candidate = _report(completeness=0.5, passed=1.0, tokens=1)

    comparison = module.compare_reports(baseline, candidate)

    assert comparison.passed is False
    assert comparison.code == "outcome_completeness_decreased"


def test_comparison_rejects_prohibited_behavior_regression():
    module = _load_script()
    baseline = _report(completeness=1.0, passed=1.0, duplicates=0)
    candidate = _report(completeness=1.0, passed=0.5, duplicates=1)

    comparison = module.compare_reports(baseline, candidate)

    assert comparison.passed is False
    assert comparison.code == "prohibited_behavior_regressed"


def test_comparison_accepts_equal_completeness_with_lower_usage():
    module = _load_script()
    comparison = module.compare_reports(
        _report(completeness=1.0, passed=1.0, tokens=100),
        _report(completeness=1.0, passed=1.0, tokens=50),
    )

    assert comparison.passed is True
    assert comparison.code == "accepted"


def test_comparison_rejects_unequal_trial_counts_before_raw_aggregates():
    module = _load_script()
    comparison = module.compare_reports(
        _report(completeness=1.0, passed=1.0, trials=2, duplicates=1),
        _report(completeness=1.0, passed=1.0, trials=4, duplicates=1),
    )

    assert comparison.passed is False
    assert comparison.code == "incompatible_trial_count"


def test_comparison_requires_matching_models():
    module = _load_script()

    with pytest.raises(ValueError, match="model"):
        module.compare_reports(
            _report(completeness=1.0, passed=1.0),
            _report(completeness=1.0, passed=1.0, model="other"),
        )


def test_command_prints_safe_json_and_uses_exit_one_for_regression(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(_report(1.0, 1.0).to_json_object()))
    candidate.write_text(json.dumps(_report(0.0, 1.0).to_json_object()))

    completed = subprocess.run(
        [sys.executable, SCRIPT, baseline, candidate],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert json.loads(completed.stdout) == {
        "code": "outcome_completeness_decreased",
        "passed": False,
    }
    assert str(baseline) not in completed.stdout + completed.stderr
    assert str(candidate) not in completed.stdout + completed.stderr


def test_command_exits_one_for_unequal_trial_counts(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(_report(1.0, 1.0, trials=2).to_json_object()))
    candidate.write_text(json.dumps(_report(1.0, 1.0, trials=4).to_json_object()))

    completed = subprocess.run(
        [sys.executable, SCRIPT, baseline, candidate],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert json.loads(completed.stdout) == {
        "code": "incompatible_trial_count",
        "passed": False,
    }


def test_command_rejects_oversized_report_without_parsing_it(tmp_path):
    oversized = tmp_path / "oversized.json"
    candidate = tmp_path / "candidate.json"
    oversized.write_bytes(b"{" + b"x" * MAX_REPORT_BYTES)
    candidate.write_text(json.dumps(_report(1.0, 1.0).to_json_object()))

    completed = subprocess.run(
        [sys.executable, SCRIPT, oversized, candidate],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert json.loads(completed.stdout) == {
        "code": "invalid_report",
        "passed": False,
    }
    assert completed.stderr == ""


def test_command_strictly_rejects_unknown_report_fields(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    payload = _report(1.0, 1.0).to_json_object()
    baseline.write_text(json.dumps(payload | {"private_transcript": "marker"}))
    candidate.write_text(json.dumps(payload))

    completed = subprocess.run(
        [sys.executable, SCRIPT, baseline, candidate],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert json.loads(completed.stdout) == {
        "code": "invalid_report",
        "passed": False,
    }
    assert "marker" not in completed.stdout + completed.stderr


def _load_script():
    loader = SourceFileLoader("compare_codex_evals", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _report(
    completeness,
    passed,
    *,
    tokens=100,
    duplicates=0,
    model="model-safe",
    trials=2,
):
    scenarios = tuple(
        ScenarioAggregate(
            name=name,
            trial_count=trials,
            passed_trials=int(passed * trials),
            complete_trials=int(completeness * trials),
            pass_rate=passed,
            completeness_rate=completeness,
            tool_counts={"agent": trials, "send_message": 0},
            input_tokens=MetricValue.observed(tokens),
            output_tokens=MetricValue.observed(tokens),
            maximum_observed_depth=MetricValue.observed(1),
            maximum_discovery_streak=1,
            reuse_calls=0,
            duplicate_launches=duplicates,
            unavailable_counts={
                "maximum_observed_depth": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            },
            failure_codes=("duplicate_agent_launch",) if duplicates else (),
        )
        for name in EVAL_SCENARIO_NAMES
    )
    return EvalReport(
        schema_version=1,
        mode="advisory",
        model=model,
        trial_count=trials,
        pass_rate=passed,
        completeness_rate=completeness,
        scenarios=scenarios,
        active_workers=MetricValue.unavailable(),
        revision_deduplication=MetricValue.unavailable(),
    )
