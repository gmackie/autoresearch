"""Evaluator for the nanochat speedrun environment.

Parses the metrics block train.py prints after the fixed 5-minute budget:

    val_bpb:          1.234567
    training_seconds: 301.2
    peak_vram_mb:     8034.1
    ...

v0.1 trusts the trainer's self-reported numbers (upstream behavior). The
hardening step — re-running prepare.evaluate_bpb on a saved checkpoint with
frozen code — is tracked in THREATS.md.
"""

import json
import math
import os
import re

TIME_BUDGET_S = 300           # prepare.py TIME_BUDGET (frozen)
TIME_SLACK = 1.10             # 10% grace on self-reported training time

FIELDS = {
    "val_bpb": "val_bpb",
    "training_seconds": "training_seconds",
    "peak_vram_mb": "peak_vram_mb",
    "mfu_percent": "mfu_percent",
    "total_tokens_M": "total_tokens_M",
    "num_steps": "num_steps",
    "num_params_M": "num_params_M",
}


def main() -> None:
    stdout_path = os.environ["AUTOLAB_CANDIDATE_STDOUT"]
    with open(stdout_path, errors="replace") as f:
        text = f.read()

    metrics: dict[str, float] = {}
    for label, metric in FIELDS.items():
        m = re.search(rf"^{re.escape(label)}:\s+([-\d.naif]+)\s*$", text, re.MULTILINE | re.IGNORECASE)
        if m:
            try:
                metrics[metric] = float(m.group(1))
            except ValueError:
                pass

    gates = {
        "run_completed": "val_bpb" in metrics,
        "val_bpb_finite": math.isfinite(metrics.get("val_bpb", float("nan"))),
        "time_budget_respected": metrics.get("training_seconds", 0.0) <= TIME_BUDGET_S * TIME_SLACK,
    }
    if not gates["val_bpb_finite"]:
        metrics.pop("val_bpb", None)

    with open(os.environ["AUTOLAB_METRICS"], "w") as f:
        json.dump({"gates": gates, "metrics": metrics}, f, indent=2)


if __name__ == "__main__":
    main()
