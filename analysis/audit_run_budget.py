"""The unit the Open Questions state their costs in.

Reads the canonical runs' recorded epoch budget and renders a single sentence
defining one run-budget, so the costs below are multiples of a measured
quantity rather than invented hours. The budget is expressed in epochs over the
full training split -- hardware-independent by construction -- so no absolute
time, and no description of the machine, enters the cost estimates. The runs
and their wall times stay in results/<run>/run.json.
"""

from __future__ import annotations

import json
from pathlib import Path

RUNS = (Path("results/unet_a/run.json"), Path("results/unet_b/run.json"))


def run() -> str:
    records = [json.loads(path.read_text()) for path in RUNS]
    epochs = {record["training"]["epochs_run"] for record in records}
    if len(epochs) != 1:
        raise ValueError(f"runs disagree on epoch budget: {sorted(epochs)}")
    budget = epochs.pop()
    return (
        f"Cost is quoted in run-budgets. One run-budget is {budget} epochs over the "
        "full training split -- the same budget both baselines used. Stating cost this "
        "way fixes the compute a question needs without fixing the time it takes, which "
        "depends on hardware this document does not describe."
    )


if __name__ == "__main__":
    print(run())
