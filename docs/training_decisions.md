# Training Decisions — CircuitNet-N14 Congestion Baseline

_Generated 2026-07-30 by `make report`. Every number below is produced by a probe
under `analysis/`, not written by hand._

_The probes require the Gold layer and exclusive use of an accelerator, so they
are not part of continuous integration and are run manually via `make probe`.
Their output is committed as JSON under `results/probes/`, and the audits that
render this document read only that JSON — this file regenerates on a machine
with neither the data nor an accelerator._

This document records the decisions taken between a training-ready Gold layer and
a trained baseline. It is a companion to `data_decisions.md` and follows the same
rule: a decision is only recorded here if a measurement supports it.

Three of the entries below reverse a conclusion this project had already reached
and, in two cases, already committed. Those reversals are kept in place rather
than tidied away. A repository whose contribution is evaluation honesty cannot
present its reasoning as having been correct from the start.

## 1. What the target actually is

The prediction target is global-routing overflow divided by track capacity. That
makes it a ratio with a small integer numerator and a per-design, per-layer
denominator, so it is not continuous: it lands on a grid of fractions dominated
by a single value.

Three consequences follow, and each one changes a decision made later in this
document. Absolute error cannot be compared across splits, because the error of
predicting zero everywhere differs between them by an order of magnitude. The
hotspot threshold cannot be placed in a gap between quantisation levels, because
on the training split alone there is no gap at the relevant scale; it is defined
physically instead, as
demand exceeding capacity by more than five percent, and the level sitting
exactly at five percent is excluded by a strict comparison. And the concentration
of squared error in the extreme tail is a property of the training split rather
than of the task, which is what rules out a magnitude-proportional loss weight
later.

```
Per split, measured over valid pixels only:

  split   patches       valid px  coverage   nonzero  zero-pred MSE      max
  train    37,704    461,483,142    0.7470    1.601%     3.0667e-05     8.00
  val       6,641     98,090,840    0.9015    0.955%     1.2987e-05    10.00
  test      3,657     51,515,921    0.8598    5.575%     1.3602e-04     0.55

The zero-predictor error spans a factor of 10.5 across splits, so an absolute error figure says nothing without the
baseline of the split it was measured on. Every error below is a ratio.

Quantisation. Targets are routing overflow over track capacity, so they
take fractional values whose denominators vary by design and metal layer:

  train   596 distinct levels
         1/44 (4,573,382), 1/22 (943,119), 3/44 (251,552), 1/43 (164,220), 1/42 (96,416), 1/11 (89,016)
  val     302 distinct levels
         1/44 (583,433), 1/22 (80,762), 1/43 (12,415), 1/27 (11,085), 1/31 (10,683), 3/44 (10,339)
  test    404 distinct levels
         1/44 (1,027,793), 1/22 (423,801), 3/44 (208,268), 1/11 (104,878), 1/43 (95,482), 1/57 (77,702)

The threshold at 0.05 is itself a level, not a gap between
levels. Neighbouring values on the training split are dense enough that no
gap exists to place it in:

  0.048780     2/41      13,903
  0.049180     3/61         387
  0.050000     1/20      12,889  <-- threshold
  0.050633     4/79           6
  0.050847     3/59         202

  pixels sitting exactly on it: train 12,889, val 1,842, test 7,112

The definition is therefore physical rather than geometric: a hotspot is a
cell whose routing demand exceeds its track capacity by more than five
percent. The level at exactly five percent is excluded by the strict
comparison, deliberately. Both round to the same single-precision value, so
the boundary does not depend on rounding -- but roughly two percent of
positive pixels sit on it, which is why evaluation sweeps thresholds
instead of reporting one.

Concentration of squared error. Each cell is the fraction of valid pixels
above the threshold, and the share of total squared error they carry:

  split               >0.01              >0.02              >0.05               >0.1               >0.2
  train     1.601% / 100.0%    1.560% /  99.6%    0.153% /  62.6%    0.026% /  41.1%    0.003% /  28.5%
  val       0.955% / 100.0%    0.891% /  98.4%    0.044% /  48.4%    0.007% /  34.5%    0.000% /  26.4%
  test      5.575% / 100.0%    5.213% /  99.1%    1.225% /  70.4%    0.230% /  30.8%    0.012% /   4.7%

On the training split 0.003% of pixels carry 28.5% of the squared error.
Squaring already amplifies large targets, so a weight proportional to the
target would compound an existing bias rather than correct one. The weighted
loss uses a binary class weight for that reason.

That concentration is a property of the training split, not of the task. On
the test split the same tail carries only 4.7% -- its error sits in the
moderate hotspot band instead, which is the band that matters in practice.

Distribution shift. The splits are design-wise, so nothing forces their
target distributions to match, and they do not: 'test' is 5.8x
denser in hotspots than 'val'.

  train  nonzero  1.601%  max   8.00  pixels at or above 1.0: 928
  val    nonzero  0.955%  max  10.00  pixels at or above 1.0: 32
  test   nonzero  5.575%  max   0.55  pixels at or above 1.0: 0

Checkpoints are selected on the validation split, which is the sparsest of
the three. Model selection therefore happens on a distribution unlike the
one the model is finally scored against. This is a limitation of the
protocol and is not corrected for.
```

## 2. Loss — masked reduction and a binary class weight

Patch coverage ranges from a sliver of a window to all of it, because edge
patches are zero-padded to a fixed size. Reducing the loss as a mean of per-patch
means would weight a pixel by the inverse of its patch's coverage, letting the
smallest edge fragments dominate. The reduction is therefore a single global masked mean
over every valid pixel in the batch. Edge fragments are kept rather than
filtered: the die boundary is where macros and IO sit, so discarding it would
discard the hard cases.

Two objectives are trained and compared. The failure mode of plain squared error
on this target is not that it ignores rare high values — the measurement in
section 1 shows squaring already amplifies them heavily. The failure is
shrinkage: where the model cannot distinguish pixels, the minimiser is the
conditional mean, which for an illustrative pixel -- hotspot with probability
0.3 and value 0.07, values chosen for the arithmetic rather than measured --
sits below any usable threshold. The model then predicts a smooth
low-amplitude field that scores well per pixel while crossing no threshold
anywhere.

The weighted variant therefore applies a binary class weight rather than one
proportional to the target, and its default solves the same conditional optimum
for the weight at which it clears the threshold. That derivation is checked
against the optimiser in `tests/test_losses.py` rather than asserted here.

## 3. Batch size and loader workers

Per-image time varies little across the batch range this accelerator can hold --
the sweep and its spread are rendered below -- and the configured size sits
within measurement drift of the fastest. The choice is made on memory headroom
instead, and for a specific reason: on this platform, exceeding device memory is
not expected to raise. The driver can fall back to host memory and the run slows
drastically with nothing in the logs to say so.

That failure was observed once, before autotuning was enabled. The records of
that episode were not retained, so it stands as testimony, and the present sweep
does not reproduce it -- autotuning selects a more frugal convolution algorithm
under pressure. The conservative batch size is kept because the failure mode it
points at is silent, not because the faster option is slower.

Loading is not the constraint and never was. An early estimate treated this task
as input-bound because compute and loading were timed together; separating them
shows loading running an order of magnitude faster than the loop can consume.

```
Compute, timed on synthetic tensors of the production shape so that loading cannot contribute:

   batch   ms/step  ms/image     peak  reserved  free after
       8      45.6      5.71    2.22G     2.26G       4.32G
      16      79.6      4.97    1.07G     1.39G       3.61G
      24     131.8      5.49    3.59G     3.65G       3.18G
      32     155.2      4.85    4.27G     4.53G       2.43G
      40     200.0      5.00    4.38G     4.50G       2.01G
      48     230.8      4.81    4.40G     4.66G       1.68G
      64     315.3      4.93    4.37G     4.85G       0.00G

  device total: 6.00 GiB

Loading, timed on real patches with no compute attached:

   workers   ms/step  ms/image   headroom
         0      34.9      1.09         5x
         2      11.8      0.37        13x
         4       5.7      0.18        28x
         8       3.5      0.11        45x

At 4 workers, loading is 28 times faster than the loop can consume. This task is bound by compute, not by input --
the opposite of what was assumed before it was measured, when the two were
timed together and loading was blamed for the step time.

Per-image time spans 4.81 to 5.71 ms across the sweep, a spread of 19% between the fastest and slowest batch size measured,
and none exceeded device memory. The spill observed earlier, without autotuning, does not reproduce here; its records were not retained.

The configured batch of 32 is therefore not chosen for speed. It runs at 4.85 ms per image against 4.93 at 64, a difference inside the noise of this measurement,
and leaves 2.43 GiB free where 64 leaves 0.00 GiB. Headroom is the whole reason: on this platform,
exceeding device memory is not expected to raise. The driver can fall back
to host memory and the run slows drastically and silently, with nothing in
the logs to say so. A configuration that cannot fail that way is worth more
than a speed difference within measurement drift.

The device was verified idle by repeating one batch size at the start and end of the compute sweep: the two agreed within 3.6%. The loading numbers are not bracketed by this check, a known limitation of the instrument. The check exists because reading free memory does not work here --
an earlier version of this probe saw most of a busy device as free, produced timings far from the truth, and slowed the training run it was competing with. The records of that episode were not retained.
```

## 4. Numerical precision

Training runs in single precision, and the reason recorded here is not the reason
originally given -- nor the first replacement for it, which was withdrawn in
turn.

Two earlier claims — that half precision produced non-finite values on the first
forward pass, and that squared errors reached the bottom of its dynamic range —
were withdrawn. The records behind them were not retained, so whether the
observation was real cannot be established; the mechanism it described is
arithmetically plausible for the head initialisation the model had at the time,
but plausibility is not a measurement. The replacement claim, an activation
ceiling with ample headroom, came from a probe that measures the model at
initialisation, where a zero-initialised head makes every examined error a
squared target; it was withdrawn for that reason, and the range question stays
open until re-measured from a trained checkpoint.

The justification that survives is throughput: both half-width formats run the
same training step materially slower than single precision on this machine, and
the timings — which depend on shapes, dtypes and kernel selection rather than on
weights, and therefore transfer to training — are rendered below. Bfloat16 is
measured alongside as a control rather than a candidate; on a zero-output model
finiteness discriminates nothing, so the control becomes informative only once
the measurements come from a trained checkpoint.

```
Identical training steps under each precision mode:

  mode        ms/image  vs fp32     peak  vs fp32  all finite
  float32         5.15    1.00x    2.06G    1.00x         yes
  float16        24.37    4.73x    2.86G    1.39x         yes
  bfloat16       24.88    4.83x    1.05G    0.51x         yes

Half precision costs 4.7 times the step time. On this machine the narrower format buys no arithmetic
throughput, and the slowdown is consistent with overhead from casting and
from the loss scaler rather than with any gain being available.

Bfloat16 lands at 4.8 times as well. Two formats with different mantissa widths agreeing this closely is
what a hardware property looks like, rather than a quirk of one of them.

The peak allocation column is not usable evidence and is shown only for completeness. Both reduced formats store activations at half the width, so both should land near 0.5x;
bfloat16 does at 0.51x while half precision reports 1.39x. The difference is autotuning scratch space, which varies with dtype in ways that
have nothing to do with the format's width -- the same effect makes peak allocation non-monotonic in batch size in the throughput probe. The case for
single precision rests on the timings alone.

The stability and range blocks, labelled for what they measure:

  largest activation, model at initialisation  5.3  (from decoders.3.block.0)
  half precision finite ceiling                65504

  squared errors examined                      7,116,839 valid pixels
  half precision smallest normal               6.10e-05
  fraction falling below it                    0.0%
  smallest non-zero squared error seen         1.88e-04

These are measurements of the model at initialisation. The output head is
zero-initialised, so the network emits exact zeros and every squared error
examined above is a squared target; the activation range is a property of
initialisation, taken in eval mode on random input. Nothing here supports a
claim about numerical behaviour during training, in either direction: not
the withdrawn claim that half precision overflows, and not the withdrawn
claim of comfortable headroom. The question stays open until these numbers
are re-taken from a trained checkpoint.

An earlier note said half precision produced non-finite values before the
head was zero-initialised. The records behind that observation were not
retained; the mechanism it described is arithmetically plausible for a
Kaiming-initialised single-channel head, but plausibility is not a
measurement.

Bfloat16 is included as a control. It carries the exponent range of single
precision with 7 mantissa bits against half precision's 10, so once these
measurements come from a trained checkpoint, a failure of range would show
up in one format and not the other. On a zero-output model both are finite
by construction, which discriminates nothing.

The device was verified idle by repeating the single-precision measurement at the start and end of the run: the two agreed within 0.3%.
```

## 5. Learning rate, and fixing the instrument first

This section exists in its current form because the probe behind it kept being
wrong.

Three invocations of the same learning-rate sweep disagreed with each other by
more than the learning rates disagreed among themselves. Under autotuned kernel
selection, a fixed seed does not produce a fixed result, and the variation was
large enough to flip a verdict between "fitted" and "collapsed" rather than to
move the last digits. Every conclusion drawn from those runs was noise, including
one already written into a commit message; the records of those invocations were
not retained.

The probe now disables autotuning, enforces deterministic kernels, and verifies
that choice by repeating a configuration and comparing losses bit for bit before
it sweeps. That costs a multiple of the runtime, which is the
right trade for something whose only job is to measure. The training entry point
deliberately does the opposite: it keeps autotuning for the throughput and
records in every run that its results are reproducible in distribution rather
than bitwise.

With the instrument fixed, spread across seeds means sensitivity to
initialisation rather than platform noise; what the earlier sweeps reported, and
that their records were not retained, is stated in the rendered verdict below.

```
Determinism, verified before sweeping:

  repeated 100 steps at lr 3e-04, seed 42
  loss, first run    1.2914424588e-05
  loss, second run   1.2914424588e-05
  identical          True

  autotuning off, deterministic kernels enforced

Without this the sweep below would be unreadable. Earlier invocations under
autotuning disagreed with themselves at a fixed seed by more than the rates
being compared disagreed with each other; their records were not retained, so
that disagreement survives as testimony, not as a figure to quote.

Fitting 32 patches, 800 steps, 3 seeds per rate. Loss is reported against the
zero-predictor baseline of 1.5062e-05 on this subset, which contains 82 hotspot pixels:

      rate  collapsed   median     best    worst   spread   recall
     3e-05 0/3          0.2164   0.1780   0.2674     1.5x    0.805
     1e-04 0/3          0.0337   0.0230   0.9298    40.3x    0.963
     3e-04 0/3          0.0413   0.0258   0.0445     1.7x    1.000  <-- configured
     1e-03 0/3          0.1889   0.1688   0.2526     1.5x    0.902
     3e-03 3/3          0.9289   0.9186   0.9690     1.1x    0.000

A ratio near zero means the subset was fitted. Optimisation is therefore
healthy: True. There is no gradient bug and no shortage of capacity, so underfitting on the full split is a question of
generalisation or of the stopping rule, not of the optimiser. This is the
cheapest diagnostic in the toolbox and it is almost never run.

At 3e-03 every seed collapsed into the zero-predictor solution and stayed
there, with no prediction anywhere above the hotspot threshold. This is what
a target whose valid train pixels are 98.4% zeros does to
optimisation: the trivial solution is a strong attractor, and a step large
enough to land in it does not leave.

At 1e-04 nothing collapsed outright, but the seeds span 40x -- from 0.0230 to 0.9298 of the baseline.
Under determinism that spread is sensitivity to initialisation, not noise.
A rate whose outcome depends on where the weights started is not a rate to
build on, however good its median looks.

That leaves 3e-04: no collapse, and repetitions within a small factor.
The configured rate of 3e-04 is among them, holding within 1.7x across seeds with a median recall of 1.000 on the subset.

The earlier, non-deterministic sweeps ranked these rates the other way
around -- 1e-4 stable, this one marginal. Their records were not retained,
so that reversal is testimony rather than evidence; what stands on its own
is that an instrument disagreeing with itself by more than the effect size
cannot rank rates, and this one no longer does.

Three seeds bound sensitivity to initialisation loosely and no more. A rate that
passes here can still fail on a fourth seed, and nothing in this table says
otherwise. Neither does fitting 32 patches predict the full split: mini-batch noise over 37,704 patches is a different optimisation problem, and the two
completed runs at the configured rate did not collapse.
```

## 6. Checkpoint selection — where the pixel metric does real damage

Sections 1 through 5 concern configuration. This one concerns the central claim
of the project, and it is the strongest evidence in it.

A training run produces one trajectory and several defensible ways to pick a
model from it. Across every run recorded here, no two rules pick the same epoch,
and the gap between them is not marginal. Selecting on validation error
repeatedly chooses a model that detects almost nothing, while a model appearing
tens of epochs later in the same run detects an order of magnitude more. In one
run the error rule chose a model whose hotspot F1 was exactly zero while a model
in the same trajectory detected hotspots.

More pointedly: the best detector in a run is frequently one that the pixel
metric rates as worse than predicting zero everywhere. Nothing in a loss curve
indicates that this is happening. The curve descends smoothly throughout.

This is why three selection rules are recorded per run instead of one, and why
checkpoints are also written on a fixed interval. Reporting a single rule would
report a choice rather than a result.

```
unet_a
  loss=masked_mse  epochs=60  zero-predictor baseline=1.2987e-05
  selected by              epoch  error/baseline       F1   recall
  lowest error                 5          0.9490   0.0022   0.0011
  lowest weighted error       18          0.9918   0.0267   0.0150
  highest hotspot F1          41          1.0007   0.0356   0.0213
  final epoch                 60          0.9835   0.0233   0.0129
  -> the F1 rule's choice scores 15.9x higher on F1, 36 epochs later
  -> that choice is rated worse than predicting zero everywhere by the pixel metric

unet_b
  loss=masked_weighted_mse  epochs=60  zero-predictor baseline=1.2987e-05
  selected by              epoch  error/baseline       F1   recall
  lowest error                 9          1.1868   0.0503   0.0404
  lowest weighted error        5          1.1905   0.0488   0.0355
  highest hotspot F1           7          1.2957   0.0740   0.0754
  final epoch                 60          1.3121   0.0658   0.0801
  -> the F1 rule's choice scores 1.5x higher on F1, 2 epochs earlier
  -> that choice is rated worse than predicting zero everywhere by the pixel metric

superseded/unet_a_no_f1_checkpoint  [superseded]
  loss=masked_mse  epochs=60  zero-predictor baseline=1.2987e-05
  selected by              epoch  error/baseline       F1   recall
  lowest error                 5          0.9484   0.0008   0.0004
  lowest weighted error       13          1.0013   0.0191   0.0104
  highest hotspot F1          44          1.0306   0.0367   0.0229
  final epoch                 60          0.9978   0.0297   0.0174
  -> the F1 rule's choice scores 44.3x higher on F1, 39 epochs later
  -> that choice is rated worse than predicting zero everywhere by the pixel metric

superseded/unet_a_patience12  [superseded]
  loss=masked_mse  epochs=18  zero-predictor baseline=1.2987e-05
  selected by              epoch  error/baseline       F1   recall
  lowest error                 6          0.9472   0.0000   0.0000
  lowest weighted error       11          0.9796   0.0126   0.0066
  highest hotspot F1          16          0.9935   0.0235   0.0131
  final epoch                 18          0.9596   0.0129   0.0067
  -> the error rule selected a model with an F1 of exactly zero, 10 epochs before the F1 rule's choice

superseded/unet_b_patience12  [superseded]
  loss=masked_weighted_mse  epochs=20  zero-predictor baseline=1.2987e-05
  selected by              epoch  error/baseline       F1   recall
  lowest error                 8          1.1795   0.0421   0.0341
  lowest weighted error        7          1.2001   0.0588   0.0491
  highest hotspot F1          16          1.3912   0.0694   0.0936
  final epoch                 20          1.2682   0.0636   0.0678
  -> the F1 rule's choice scores 1.7x higher on F1, 8 epochs later
  -> that choice is rated worse than predicting zero everywhere by the pixel metric

Across 5 runs the two rules never select the same epoch.
In the 4 run(s) long enough for the ratio to mean anything, the F1 rule's choice scores between 1.5x and 44.3x higher on F1.
In 1 run(s) the error rule selected a model that detected no hotspot at all, while a model in the same run did.
Ratios from runs shorter than 15 epochs are shown per run but excluded from that range: an F1 near zero makes the denominator unstable.
Selecting on pixel error is therefore not a neutral default. It is a choice that discards most of the run's detection performance, and the loss curve gives no indication that it is happening.
```

## 7. Stopping rule

Early stopping was removed after it was observed selecting on noise. Validation
error fluctuates around a flat level from the first epoch while the training
objective is still descending, so a patience counter measures the fluctuation and
halts at an arbitrary point. In the superseded runs it chose a noise minimum for
the plain objective and cut the weighted run short while its hotspot recall was
still climbing.

No validation metric available at this scale gives a usable stopping signal, so
the runs use a fixed epoch budget and record the whole trajectory. That budget is
a time constraint, not a convergence criterion, and each run record says so.

## Process

Every retraining of the baseline was caused by a measurement that contradicted a
standing assumption: first early stopping selecting on noise, then
a selection rule that only became interesting after the run had finished and for
which no checkpoint existed. The runs that were replaced are kept under
`results/superseded/` with their full trajectories, because the argument in
section 6 rests on them.

Model weights are excluded from the repository. Every run record therefore stores
a SHA-256 per checkpoint, the git revision, and whether the worktree was clean —
without those, a reported number has no link to the model that produced it.

## Limitations

- Every result here comes from a single seed per configuration. No claim is made
  about variance across seeds, and the selection gap in section 6 has not been
  measured against seed variation.
- No model output has touched the test split, but two of its target statistics
  have been computed and were cited in design rationales: the count of pixels
  sitting exactly on the threshold, in the argument for evaluating a range of
  thresholds, and the distinct-level count, in the argument that no threshold
  sits in a gap. Both rationales are now anchored to the training split, with
  the cross-split spread kept as corroboration; this note is the disclosure.
- Checkpoints are selected on the validation split, which is the sparsest of the
  three by a wide margin. Model selection therefore happens on a distribution
  unlike the one the model is finally scored against. This is not corrected for.
- The learning-rate sweep fits thirty-two patches full-batch. Mini-batch noise
  over the full training split is a different optimisation problem, so a rate
  that is stable on the subset is not thereby stable on the whole.
- Three seeds bound sensitivity to initialisation loosely. A rate that passes the
  stability check can still fail on a fourth seed.
- Throughput and memory figures describe one accelerator. The silent spill in
  section 3 is a property of this platform's driver behaviour and may not
  transfer.
- The epoch budget was chosen to fit two runs in one session. Neither run had
  converged when it stopped; the training objective was still descending.