# Data Engineering Decisions — CircuitNet-N14 Congestion Pipeline

_Generated 2026-07-25 by `make report`. Every number below is produced from the
pipeline manifests, not written by hand. To reproduce from scratch:
`make acquire bronze silver gold && make report`._

This document records the decisions taken from raw acquisition to the
training-ready Gold layer. Each one is justified by measured evidence rather than
convention. The contribution of this project is not the model but the data rigor:
every reduction, merge, and reshape below is defended by a number that can be
regenerated.

## 1. Acquisition — which dataset, which subset

CircuitNet-N14 (14nm) was chosen over N28: it is hosted on Hugging Face with a
resumable, checksummed download path, whereas N28 is distributed via Google Drive.
Only the routability (congestion) subset was downloaded, for 6 of the 8 available
designs, pinned to an immutable commit revision.

The two small-variant designs (`nvdla-small`, `Vortex-small`) were excluded
deliberately: they share a design family with chips already assigned to a split,
and keeping them would leak family-level characteristics across train/val/test.

```
repo:      CircuitNet/CircuitNet (dataset)
revision:  df74f33e302f927bf0548dc0d7936ca43b634f2a
files:     6 design tarballs
total raw: 23.9 GB (compressed)

design split:
  test   openc910-1
  train  RISCY-FPU, RISCY, nvdla-large, zero-riscy
  val    Vortex-large
```

## 2. Deduplication — the dataset is smaller than it looks

SHA-1 fingerprinting at the Silver layer showed that most samples are
bit-identical duplicates of one another. The PnR sweep parameters `fi` and `mp`
appear in the filenames but do not alter the routability feature maps, so many
nominally distinct configurations produce identical data.

Reporting the raw sample count would overstate the dataset. The honest figure is
the unique count. Importantly, no duplicate group spans more than one design, so
the design-wise split remains leak-free even before deduplication.

```
total samples:           10606
unique (by sha1):        4216
duplicate rate:          60%
samples in a dup group:  10280
dup groups cross-design: 0  (0 = design-wise split is leak-free)

per design (total -> unique):
  RISCY           3456 -> 1318
  RISCY-FPU       3456 -> 1336
  Vortex-large      74 ->   65
  nvdla-large       68 ->   68
  openc910-1        96 ->   91
  zero-riscy      3456 -> 1338
```

## 3. Ground-truth sparsity — why pixel metrics mislead

The ground truth is the per-pixel maximum of horizontal and vertical global-routing
overflow. Horizontal and vertical overflow were measured to be nearly uncorrelated
and asymmetric (vertical dominates), so they are kept separate at the Bronze layer
and merged only at Silver — the merge strategy can be revisited without re-ingesting.

The merged GT is extremely sparse, and this is the central evaluation argument of
the project: a trivial all-zero predictor scores near-perfectly on pixel-level
metrics while detecting no hotspots at all. This is what motivates a
domain-specific hotspot metric later in the project rather than reporting SSIM or
NRMS alone.

```
unique samples:          4216
gt_frac_nonzero mean:    0.0179  (1.8% of pixels are hotspots)
gt_frac_nonzero min/max: 0.0023 / 0.1530

An all-zero predictor is correct on ~98.2% of pixels,
scoring near-perfect on pixel metrics (SSIM/MSE/NRMS) while detecting
zero hotspots — the only thing that matters for routability.
```

## 4. Shape handling — crop, do not resize

Raw feature maps have variable shape, and the shape clusters tightly by chip:
small CPU designs are roughly 250 px across while the largest design approaches
1640 px. Resizing everything to a fixed square would cause two problems. First,
congestion is a density defined on a physical routing grid, so rescaling distorts
the quantity being predicted. Second, and more seriously, it would confound the
design-wise domain shift with a resize-induced loss of detail: training chips
would be barely rescaled while validation and test chips would be downscaled
several-fold, making a poor test score impossible to attribute.

Chips are therefore cropped into non-overlapping 128x128 patches at native
resolution. Patch size 128 was chosen over 256 because it requires no padding for
the smallest chips (256 would pad roughly three quarters of them) and yields a
substantially larger training set. Edge patches are zero-padded to a full 128x128
and accompanied by a valid mask, so padded pixels can be excluded from both the
loss and the evaluation metrics.

```
unique samples: 4216
H range: 167-1731 (median 225)
W range: 228-1729 (median 320)
aspect W/H median: 1.45  (near-square: 37%)

per-design median shape (size clusters tightly by chip):
  RISCY          n= 1318  215x312
  RISCY-FPU      n= 1336  230x334
  Vortex-large   n=   65  1228x1202
  nvdla-large    n=   68  1642x1638
  openc910-1     n=   91  738x736
  zero-riscy     n= 1338  210x296
```

## 5. Patching and normalization

The `cell_density` and `rudy` channels are z-scored using statistics fit on
training valid pixels only, so no validation or test statistics leak into the
transform. `macro_region` is already binary and is left untouched. The ground
truth is left in its raw scale so that reported scores remain comparable to
published baselines. The valid mask travels with every patch for masked loss and
masked metrics downstream.

```
patch size:            128x128
total patches:         48002
  train / val / test:  37704 / 6641 / 3657
fully-valid patches:   27617
edge patches (padded): 20385 (42%)

normalization (z-score, fit on TRAIN valid pixels only):
  cell_density   mean=1.020202 std=0.974116
  rudy           mean=0.007696 std=0.007890
  macro_region:  left binary (not z-scored)
  gt:            left raw (target, comparable to literature)
```

## Limitations

- Only 6 independent chip designs exist. The large per-design sample counts are
  parameter sweeps of the same chip, so generalization is measured at the design
  level from very few independent points.
- Validation and test each rest on a single design. Their metrics therefore carry
  higher variance and measure transfer to one specific chip rather than to unseen
  chips in general.
- A 128x128 patch gives the model a local receptive field. Congestion driven by
  global layout structure beyond a single patch is not captured by this framing.
- The horizontal/vertical merge uses a per-pixel maximum. This follows common
  practice but discards directional information; the separate maps are retained
  at Bronze so an alternative merge can be evaluated without re-ingesting.