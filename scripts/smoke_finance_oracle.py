"""Run known-answer oracle and failure baselines without Ollama or Inspect."""

from pathlib import Path

from main_module.finance_harness.core import InvoiceSandbox

root = Path(__file__).resolve().parents[1]
case = root / "cases" / "finance" / "smoke_correction.json"

for name in ("oracle", "always_release"):
    env = InvoiceSandbox.from_path(case)
    env.prepare_payment(480_000)
    env.apply_world_event(env.initial["event"])
    if name == "oracle":
        env.request_approvals(["budget_owner", "finance_manager"])
    else:
        env.release_payment(480_000)
    result = env.outcome()
    path = env.save_trace(root / "runs" / f"{name}.jsonl")
    print(f"{name}: {result} | trace: {path}")
    assert result == ("acceptable" if name == "oracle" else "invalid_payment")
