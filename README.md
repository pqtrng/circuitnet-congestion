# Routing Congestion Prediction on CircuitNet-N14

A data-engineering-first pipeline for predicting global-routing congestion from placement-stage layout features, built
around a single question: **what does a model evaluation on this task actually measure?**

The model is an ordinary U-Net. The contribution is the measurement discipline around it — a reproducible data pipeline
with fingerprinted lineage, and an evaluation that reports where its own metrics mislead.

## Status

Phase A (data pipeline) and Phase B (baseline training) are complete. No model has been evaluated on the test split;
its target statistics have, however, been computed and are used in two design rationales recorded in the decision
documents. Every model-derived number below is from validation. Test evaluation is the next task.

## The finding

On this target, the metric used to select a checkpoint changes the resulting model more than the loss function does.

| run | selected by | epoch | pixel error / zero-predictor | hotspot precision | recall | F1 |
|---|---|---|---|---|---|---|
| predict zero everywhere | — | — | 1.000 | — | 0.0000 | 0.0000 |
| plain squared error | lowest error | 5 | 0.949 | 0.2160 | 0.0011 | 0.0022 |
| plain squared error | highest F1 | 41 | **1.001** | 0.1097 | 0.0213 | 0.0356 |
| weighted | lowest error | 9 | **1.187** | 0.0667 | 0.0404 | 0.0503 |
| weighted | highest F1 | 7 | **1.296** | 0.0727 | 0.0754 | 0.0740 |

The reference row predicts zero at every pixel: on the validation split that leaves 99.0% of valid pixels exactly right while finding none of the 0.044% that are hotspots. Bold marks a pixel error above it -- the metric rating a trained model worse than predicting nothing.

The two rules never pick the same epoch, in any run recorded. In the plain run they land tens of epochs apart. The
reference row is the reason this matters: across the whole run the pixel metric never separates a trained model from
predicting nothing by more than a narrow margin either way — the best it rates lands a few percent below that
predictor, the model that actually detects the most hotspots a hair above it. The entire trajectory the metric sees
is a band around a useless predictor, and the loss curve descends smoothly through all of it, giving no sign that the
selection underneath is turning over. Read the same rows by hotspot recall instead and the runs are an order of
magnitude apart.

Precision and recall sit beside F1 because a single F1 hides which half fails, and the two runs fail differently: the
plain run holds the table's highest precision while its recall collapses to almost nothing, so it is not detecting
hotspots so much as declining to guess. The weighted run trades that for balanced but low detection on both sides.

This is not a subtle effect. It follows directly from the target, measured below before any model exists: on every
split predicting zero is right almost everywhere, because almost nowhere is a hotspot.

Absolute F1 here is low, and no attempt is made to present it otherwise. The result is the *shape* of the disagreement,
not the height of the bars.

## Why the target behaves this way

Ground truth is global-routing overflow divided by track capacity. That makes it a ratio with a small integer numerator
and a per-design, per-layer denominator — not a continuous field but a grid of fractions, dominated by a single value.

Validation rests on a single design (Vortex-large) and test on another (openc910-1); training holds the rest.

| split | valid pixels | zero-predictor correct | pixels that are hotspots |
|---|---|---|---|
| train | 461,483,142 | 98.4% | 0.153% |
| val | 98,090,840 | 99.0% | 0.044% |
| test | 51,515,921 | 94.4% | 1.225% |

A metric can be almost perfect and useless at the same time, and the two columns above are how: predicting zero scores
well on every split while finding none of the hotspots, which are the entire point of the task. This is why the project
reports a domain hotspot metric rather than pixel error alone, and why absolute error is not comparable between splits
in the first place -- the fraction that is a hotspot, and so the error there is to make, differs across them.

| split | patches | non-zero pixels | zero-predictor error |
|-------|---------|-----------------|----------------------|
| train | 37,704 | 1.601% | 3.07e-05 |
| val | 6,641 | 0.955% | 1.30e-05 |
| test | 3,657 | 5.575% | 1.36e-04 |

The zero-predictor error spans a factor of 10.5 across splits, so **absolute error is not comparable between them**.
Every figure in this repository is reported as a ratio against the baseline of its own split.

Full measurement: [`docs/data_decisions.md`](docs/data_decisions.md) and
[`docs/training_decisions.md`](docs/training_decisions.md).

## Method

**Data.** CircuitNet-N14 routability subset, 6 of 8 designs, pinned to an immutable revision. Bronze/Silver/Gold layers
with Pandera schema validation and SHA-1 fingerprinting. Deduplication collapsed the nominal sample count to well under
half — the fingerprints live in `data/silver/manifest.json` and the counts are rendered in `docs/data_decisions.md` —
because the sweep parameters in the filenames do not alter the feature maps. No duplicate group spans designs, so the
split stays leak-free.

**Splits are design-wise**: training chips share no design family with validation or test. Small variants of designs
already assigned to a split were excluded rather than reused.

**Patches** are 128×128 crops at native resolution, never resized: congestion is a density on a physical routing grid,
and rescaling would confound domain shift with resolution loss. Edge patches are zero-padded and carry a valid mask,
which travels through the loss and every metric. Normalisation statistics are fitted on the training split alone.

**Model.** U-Net, depth 4, 7.8M parameters, bilinear upsampling, linear output head. Trained in single precision for 60
epochs at a fixed learning rate.

**Configuration constants are measured where a measurement exists, and disclosed as choices where it does not** — the
loader worker count, for one, is a choice inside a region a measurement showed to be insensitive. Batch size, precision
and learning rate each have a probe under `analysis/` that produces the evidence, and `docs/training_decisions.md`
records the reasoning — including cases where a measurement reversed a decision this project had already made and
committed.

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
revision and whether the worktree was clean, so a reported number can be traced to the weights that produced it -- and
the weights themselves are published in the [`model-v1`](../../releases/tag/model-v1) release, five files covering every
selection rule that still has them, verifiable against those digests with `sha256sum -c SHA256SUMS`. What is not there:
`unet_a`'s weighted-error selection, whose checkpoint write was gated off at the time, and every selection in the three
superseded runs, whose files were deleted after their digests were recorded.
`docs/training_decisions.md` renders which selections are loadable and which survive as metrics only.

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
| `tests/` | CPU-only test suite, synthetic fixtures |

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

`docs/training_decisions.md` states, for each unresolved point, what would settle it and what it would cost in
run-budgets: whether half precision fails on a trained checkpoint, whether the selection gap survives seed variation,
whether the learning-rate ranking holds at full-batch scale, and what the unevaluated test split says. Two further
design questions live only here:

- Does a threshold-aware objective close the selection gap, or only move it?
- The horizontal/vertical merge takes a per-pixel maximum, following common practice but discarding direction. Both maps
  are retained at Bronze so an alternative merge can be evaluated without re-ingesting.

## Data

CircuitNet is released by the CircuitNet authors; see the dataset card for licensing. This repository contains no
dataset content — only manifests, checksums and derived statistics.