# Routing Congestion Prediction on CircuitNet-N14

A data-engineering-first pipeline for predicting global-routing congestion from placement-stage layout features, built
around a single question: **what does a model evaluation on this task actually measure?**

The model is an ordinary U-Net. The contribution is the measurement discipline around it — a reproducible data pipeline
with fingerprinted lineage, and an evaluation that reports where its own metrics mislead.

## Status

Phase A (data pipeline) and Phase B (baseline training) are complete. The test split has not been evaluated and is
untouched by any decision recorded here; every number below is from validation. Test evaluation is the next task.

## The finding

On this target, the metric used to select a checkpoint changes the resulting model more than the loss function does.

| run                 | selected by  | epoch | val error / zero-predictor | hotspot F1 |
|---------------------|--------------|-------|----------------------------|------------|
| plain squared error | lowest error | 5     | 0.949                      | 0.0022     |
| plain squared error | highest F1   | 41    | **1.001**                  | **0.0356** |
| weighted            | lowest error | 9     | 1.187                      | 0.0503     |
| weighted            | highest F1   | 7     | 1.296                      | **0.0740** |

The two rules never pick the same epoch, in any run recorded. In the plain run they land 36 epochs apart and differ by a
factor of 16 in hotspot F1. The model that detects hotspots best is one that squared error rates as *worse than
predicting zero everywhere* — and the loss curve descends smoothly throughout, giving no indication that this is
happening.

This is not a subtle effect. It follows directly from the target: 98.5% of valid pixels are zero, so a predictor that
outputs nothing achieves a low error, and a model is rewarded for staying near it.

Absolute F1 here is low, and no attempt is made to present it otherwise. The result is the *shape* of the disagreement,
not the height of the bars.

## Why the target behaves this way

Ground truth is global-routing overflow divided by track capacity. That makes it a ratio with a small integer numerator
and a per-design, per-layer denominator — not a continuous field but a grid of fractions, dominated by a single value.

| split | design       | patches | non-zero pixels | zero-predictor error |
|-------|--------------|---------|-----------------|----------------------|
| train | 4 designs    | 37,704  | 1.601%          | 3.07e-05             |
| val   | Vortex-large | 6,641   | 0.955%          | 1.30e-05             |
| test  | openc910-1   | 3,657   | 5.575%          | 1.36e-04             |

The zero-predictor error spans a factor of 10.5 across splits, so **absolute error is not comparable between them**.
Every figure in this repository is reported as a ratio against the baseline of its own split.

Full measurement: [`docs/data_decisions.md`](docs/data_decisions.md) and
[`docs/training_decisions.md`](docs/training_decisions.md).

## Method

**Data.** CircuitNet-N14 routability subset, 6 of 8 designs, pinned to an immutable revision. Bronze/Silver/Gold layers
with Pandera schema validation and SHA-1 fingerprinting. Deduplication reduced 10,606 nominal samples to 4,216 unique
ones — the sweep parameters in the filenames do not alter the feature maps. No duplicate group spans designs, so the
split stays leak-free.

**Splits are design-wise**: training chips share no design family with validation or test. Small variants of designs
already assigned to a split were excluded rather than reused.

**Patches** are 128×128 crops at native resolution, never resized: congestion is a density on a physical routing grid,
and rescaling would confound domain shift with resolution loss. Edge patches are zero-padded and carry a valid mask,
which travels through the loss and every metric. Normalisation statistics are fitted on the training split alone.

**Model.** U-Net, depth 4, 7.8M parameters, bilinear upsampling, linear output head. Trained in single precision for 60
epochs at a fixed learning rate.

**Every configuration constant is measured**, not conventional. Batch size, worker count, precision and learning rate
each have a probe under `analysis/`
that produces the evidence, and `docs/training_decisions.md` records the reasoning — including three cases where a
measurement reversed a decision this project had already made and committed.

## Reproducing

```bash
make setup                    # accelerator detected automatically
make acquire bronze silver gold
make train CONFIG=configs/unet_a.yaml
make report                   # regenerate both decision documents
```

The raw download is roughly 24 GB and the intermediate layers need about 40 GB free. Manifests and probe records are
committed as evidence, so `make report`
regenerates both documents on a fresh clone with no data and no accelerator.

`make probe` re-runs the measurement suite. It needs the Gold layer and exclusive use of an accelerator, and is not part
of continuous integration.

Model weights are excluded from the repository. Every run record stores a SHA-256 per checkpoint alongside the git
revision and whether the worktree was clean, so a reported number can be traced to the weights that produced it.

## Layout

| path | contents |
|---|---|
| `src/circuitnet_congestion/data/` | acquisition, Bronze, Silver, Gold |
| `src/circuitnet_congestion/models/` | U-Net |
| `src/circuitnet_congestion/training/` | dataset, losses, metrics, training loop |
| `analysis/` | `probe_*.py` measure, `audit_*.py` render committed JSON |
| `configs/` | data and training configuration |
| `docs/` | decision records, generated by `make report` |
| `results/` | run histories, probe records, superseded runs |
| `tests/` | 115 tests, CPU-only, synthetic fixtures |

Probes need artefacts that are not in the repository; audits read only committed
JSON. That separation is what lets the documents regenerate anywhere.

## Limitations

- **The test split has not been evaluated.** Nothing here is a generalisation claim.
- Every result comes from a single seed. No claim is made about seed variance, including for the selection gap above.
- Checkpoints are selected on validation, which is the sparsest split by a wide margin — model selection happens on a
  distribution unlike the one the model will be scored against. This is not corrected for.
- Only 6 independent designs exist. Validation and test rest on one design each, so generalisation is measured from very
  few independent points.
- Neither run had converged at 60 epochs; the training objective was still descending. The budget was a time constraint.
- A 128×128 patch gives a local receptive field. Congestion driven by global layout structure beyond one patch is
  outside this framing.
- Throughput and memory measurements describe one accelerator.

## Open questions

- Does the selection gap survive across seeds, and how wide is its variance?
- Does a threshold-aware objective close the gap, or only move it?
- The horizontal/vertical merge takes a per-pixel maximum, following common practice but discarding direction. Both maps
  are retained at Bronze so an alternative merge can be evaluated without re-ingesting.

## Data

CircuitNet is released by the CircuitNet authors; see the dataset card for licensing. This repository contains no
dataset content — only manifests, checksums and derived statistics.