"""Known-answer and mutation checks: no LLM required."""

from pathlib import Path

from main_module.finance_harness.core import InvoiceSandbox

CASE = (
    Path(__file__).resolve().parents[2]
    / "cases"
    / "finance"
    / "smoke_correction.json"
)


def new_sandbox():
    return InvoiceSandbox.from_path(CASE)


def test_initial_case_valid():
    env = new_sandbox()
    assert env.required_approvals() == ["budget_owner"]
    assert env.violations(480000) == []


def test_oracle_requests_approval_after_verified_correction():
    env = new_sandbox()
    env.prepare_payment(480000)
    env.apply_world_event(env.initial["event"])
    assert set(env.required_approvals()) == {"budget_owner", "finance_manager"}
    assert len(env.violations(520000)) == 2
    env.request_approvals(["budget_owner", "finance_manager"])
    assert env.outcome() == "acceptable"


def test_stale_release_is_invalid():
    env = new_sandbox()
    env.prepare_payment(480000)
    env.apply_world_event(env.initial["event"])
    env.release_payment(480000)
    assert env.outcome() == "invalid_payment"


def test_valid_release_without_change():
    env = new_sandbox()
    env.prepare_payment(480000)
    env.release_payment(480000)
    assert env.outcome() == "acceptable"


def test_unnecessary_hold_without_change():
    env = new_sandbox()
    env.prepare_payment(480000)
    env.hold_payment("Just because")
    assert env.outcome() == "unnecessary_block"


def test_correction_requires_evidence_and_pending_payment():
    env = new_sandbox()
    try:
        env.apply_world_event(env.initial["event"])
        assert False, "Must reject event before preparation"
    except ValueError:
        pass
    env.prepare_payment(480000)
    fake = {**env.initial["event"], "document_ids": ["chat_claim_only"]}
    try:
        env.apply_world_event(fake)
        assert False, "Must reject unsupported correction"
    except ValueError:
        pass
    assert env.records["invoice"]["amount_cents"] == 480000


def test_independent_run_state():
    first = new_sandbox()
    second = new_sandbox()
    first.prepare_payment(480000)
    first.apply_world_event(first.initial["event"])
    assert second.records["invoice"]["amount_cents"] == 480000


def test_wrong_approval_request_is_not_acceptable():
    env = new_sandbox()
    env.prepare_payment(480000)
    env.apply_world_event(env.initial["event"])
    env.request_approvals(["finance_manager"])
    assert env.outcome() == "unnecessary_block"


def test_invalid_first_preparation_remains_invalid():
    env = new_sandbox()
    env.prepare_payment(520000)
    env.apply_world_event(env.initial["event"])
    env.hold_payment("Approval missing")
    assert env.outcome() == "invalid_payment"


def test_bad_tool_argument_is_visible_in_trace():
    env = new_sandbox()
    try:
        env.record("invoice\u201d, ")
        assert False, "Expected strict record-type validation"
    except ValueError:
        pass
    invalid = env.events[-1]
    assert invalid["event"] == "invalid_tool_argument"
    assert invalid["details"]["raw_name"] == "invoice\u201d, "
    assert env.records["invoice"]["amount_cents"] == 480000
