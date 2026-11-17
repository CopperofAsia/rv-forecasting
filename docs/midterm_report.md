# Midterm Report — Data-Efficient Realized Volatility Forecasting via Cross-Stock Transfer Learning

**Course:** DDA4220 Deep Learning Project
**Team:** Jiuzhi Yang (124020410), Xuanxue Zheng (123090886), Yijian Zhang (122090776), Ziyu Zhang (122090786)
**Date:** 2026-11-18

## Project at a glance

We study whether pretraining a modest neural network on several *source* stocks
reduces the amount of *target*-stock data needed to forecast the next 30 minutes
of realized volatility. Target stocks are held out of pretraining and given only
a controlled fraction of their earlier observations for adaptation; we compare
this against models trained from scratch on exactly those observations, and
against simple statistical baselines. The headline deliverable is a test-error
versus target-data curve and an estimate of the smallest target-data fraction
that reaches within 5% of a full-data reference.

This report documents what we have built and found so far, the challenges we are
working through, our plan and timeline to the final deadline, and each member's
contribution.

## 1. What we have done

### 1.1 Data pipeline

We use the HF Data Library's one-minute U.S. equity bars, derived from IEX
Exchange historical trades. The team shares a single snapshot (downloaded
2026-11-16) so results are reproducible; we record the download date and
cleaning version because the library is corrected over time. From 200 downloaded
tickers, a liquidity audit on the library's daily quality indicators (gap rate,
observed bars per day, dollar volume) screens the universe; with thresholds fixed
after the audit (gap rate ≤ 0.30, ≥ 200 observed bars, ≥ 250 valid days, ≥ $1M
median dollar volume), **148 of 200 stocks** qualify.

For each eligible stock and decision minute we build a feature window from the
preceding 30 clock-minutes and a label from the following 30 clock-minutes. The
label is realized volatility over that future window, computed from consecutive
one-minute log returns as RV₃₀ = Σ rⱼ² and σ₃₀ = √RV₃₀. Windows are built on a
regular one-minute clock grid per session, so the horizon is defined by wall-clock
time rather than by the number of traded bars; missing minutes are never filled,
and a return spanning a gap is dropped, which enforces the "consecutive observed
prices" rule. We require at least 20 valid returns in a window and report the
per-stock discard rate.

Several leakage controls are built into the pipeline: features at a decision
minute use only bars at or before it, while the label uses only the future
window, and the two never share a bar; source and target stock sets are disjoint
(≈ 80/20 split, giving 118 source and 30 target stocks); and a single shared
calendar cutoff (2025-04-04 here) separates the training period from the
validation and test periods, so no source example from the target evaluation
period can enter pretraining through another stock. The feature standardizer is
fit only on the rows permitted to each training stage. The resulting dataset
holds ≈ 1.07 million windows.

### 1.2 Models and methods

We implemented the full comparison set from the proposal:

- **Simple baselines:** a persistence forecast (predict the most recently
  observed realized volatility), an ordinary least-squares linear regression on
  the fixed feature vector, and a HAR-style regression on strictly lagged
  daily/weekly/monthly realized-volatility summaries.
- **Target-only MLP:** a small two-hidden-layer network with a softplus output
  (so predictions are nonnegative), trained from random initialization on each
  target-data fraction.
- **Cross-stock pretraining and fine-tuning:** the same architecture is
  pretrained on the pooled source windows, then adapted per target and fraction
  by either full-parameter fine-tuning or output-layer-only fine-tuning. A
  zero-shot prediction from the pretrained model (no target adaptation) gives the
  0% reference point.

### 1.3 Experimental protocol

The main independent variable is the fraction of target training observations,
taken as nested earliest-chronological prefixes [1, 5, 10, 25, 50, 100]%, so the
same rows feed the scratch and fine-tuned models. Forecast accuracy is measured
with root-mean-squared percentage error (RMSPE) over strictly positive labels,
with zero-volatility labels excluded and counted. Data efficiency is summarized
by the smallest evaluated fraction at which a method's error falls within 1.05×
the target-only MLP trained on 100% of the target data (no interpolation; "never
meets" is a valid outcome). Hyperparameters are chosen from a small prespecified
learning-rate/epoch grid on validation data, and results are averaged over three
random seeds.

### 1.4 Preliminary results

Across all 30 target stocks and 3 seeds, the mean test RMSPE is:

| target-data fraction | persistence | linear | HAR | zero-shot | scratch MLP | fine-tune (full) | fine-tune (head) |
|---|---|---|---|---|---|---|---|
| 0%  | 0.451 | 0.353 | 0.413 | 0.426 | —     | —     | —     |
| 1%  | 0.451 | 0.353 | 0.413 | —     | 0.934 | 0.379 | 0.440 |
| 10% | 0.451 | 0.353 | 0.413 | —     | 0.920 | 0.408 | 0.449 |
| 100%| 0.451 | 0.353 | 0.413 | —     | 0.770 | 0.445 | 0.444 |

![Mean test RMSPE versus target-data fraction across the 30 target stocks, with
±1 standard-deviation bands for the fraction-dependent methods and dashed
reference lines for the flat baselines.](figures/agg_rmspe_curve.png)

*Figure 1. Mean test RMSPE versus target-data fraction (30 targets, 3 seeds).
Shaded bands are ±1 standard deviation across targets; dashed lines are the
fraction-independent baselines.*

Two findings stand out. First, **cross-stock transfer is strongly data-efficient
relative to training from scratch**: full fine-tuning lowers RMSPE by about 59%
at the 1% budget and still by ~29% at 100%, and it reaches within 5% of the
scratch-at-100% reference using just 1% of target data (meeting the criterion for
29 of 30 targets at fraction 0.01, versus fraction 1.0 for scratch). Full
fine-tuning edges out output-layer-only adaptation. Second, and against the
neural hypothesis, **the simple linear baseline (0.353) is the best method
overall**: fine-tuning beats the scratch MLP for 29 of 30 targets but beats the
best baseline for only 11 of 30. This echoes Branco et al. (2024), who found no
systematic evidence that nonlinear machine-learning models beat linear
alternatives for realized-volatility forecasting; a null or negative transfer
result relative to linear models is itself informative and is reported as such.
Figure 2 isolates the fine-tuning comparison: full-parameter fine-tuning and
output-layer-only adaptation both sit far below the scratch MLP across the whole
range, with full fine-tuning slightly better at the smallest budgets.

![Full-parameter versus output-layer-only fine-tuning against the scratch MLP,
mean test RMSPE across targets.](figures/full_vs_head.png)

*Figure 2. Fine-tuning scope: full-parameter versus output-layer-only, against
the scratch MLP (mean across targets).*

As a secondary analysis we varied the number of source stocks used in
pretraining (10, 30, 60, and all 118). The effect on target data efficiency is
essentially flat (mean RMSPE ≈ 0.39 at the 5% budget regardless of source count),
suggesting that a handful of source stocks already captures the shared intraday
volatility structure and that broader pretraining does not further improve data
efficiency in this setting.

![Mean test RMSPE of full fine-tuning versus the number of source stocks used in
pretraining, for two target-data fractions.](figures/source_count.png)

*Figure 3. Effect of source-stock count on data efficiency: mean test RMSPE
(full fine-tuning) versus number of source stocks, at the 5% and 25% budgets.*

### 1.5 Engineering and reproducibility

The codebase centralizes every prespecified constant (fractions, thresholds, the
1.05× criterion, seeds) in one configuration module so the protocol is auditable.
The experiment runner is resumable and caches pretrained models, which lets the
full sweep run in chunks on CPU. A test suite (13 checks) asserts the leakage
controls, the label math, the metric behavior, and the training loop. An
independent code review of the pipeline surfaced and we fixed a baseline-scaling
bug, a window-horizon definition issue (now clock-based), and a feature-scaling
edge case; the leakage-critical machinery passed review.

## 2. Challenges we are currently facing

**Neural methods do not yet beat the simple baselines.** The linear and HAR
baselines outperform every neural variant, including fine-tuning. We need to
determine whether this reflects an under-expressive feature set, insufficient
model capacity, or a genuinely near-linear task at this horizon and data scale.
This is the most important open question for the second half of the project.

**Compute budget.** Training runs on CPU, so the full sweep (30 targets × 6
fractions × 3 methods × 3 seeds, plus source pretraining) must be executed in
resumable chunks, and we subsample the source pool to 300k rows for pretraining.
This constrains how wide a hyperparameter search we can afford.

**Data sparsity.** IEX represents only ~2–3% of consolidated U.S. volume, so many
minutes have no trades. Our clock-grid windowing and minimum-valid-returns rule
handle this without gap-filling, but they also discard some windows, and the
conclusions describe an IEX-based research dataset rather than the full market.

**Robustness check in progress.** The Raw-versus-Clean cleaning-version
comparison is implemented (including a mechanism to reuse the Clean source/target
partition so the comparison is apples-to-apples) but awaits the Raw-version
download before it can be run.

## 3. Plan and rough timeline

The midterm checkpoint is 2026-11-18 and the final deadline is 2026-12-19. Our
plan for the remaining roughly four weeks:

- **Week of Nov 18–24:** complete the Raw-versus-Clean robustness run and fold it
  in as a sensitivity section; begin diagnosing why the neural methods trail the
  linear baseline (inspect residuals, test a richer feature set and a slightly
  larger network, confirm the finding is stable across targets).
- **Week of Nov 25–Dec 1:** finalize the feature/architecture decisions from that
  diagnosis; run the consolidated main comparison with the chosen settings across
  all targets and seeds; (optional, if coverage allows) add the 10-minute horizon.
- **Week of Dec 2–8:** stability and heterogeneity analysis across target stocks;
  finalize all figures and tables; internal review of claims against the results
  files.
- **Week of Dec 9–19:** write and revise the final report, with buffer for
  re-runs; prepare any presentation materials.

If time runs short we will keep the primary 30-minute target and the main
scratch-versus-fine-tuning curve, and reduce the source-count and Raw-version
sensitivity analyses first, as stated in the proposal.

## 4. Team member contributions

The work follows the four work packages from the proposal, with one member
leading each:

| Work package | Lead | Scope |
|---|---|---|
| Literature and protocol | Yijian Zhang (122090776) | Core papers (HAR, DeepVol, intraday commonality, transfer-learning surveys); finalize horizon, splits, metrics, and the comparison grid |
| Data audit and preparation | Jiuzhi Yang (124020410) | Shared snapshot download; liquidity screening; window/label construction; leakage controls and exclusions; discard reporting |
| Models and experiments | Ziyu Zhang (122090786) | Baselines, target-only MLP, source pretraining and fine-tuning; training loop and hyperparameter grid; experiment orchestration |
| Analysis and report | Xuanxue Zheng (123090886) | Controlled comparisons; error curves and efficiency thresholds; target heterogeneity and source-count analysis; limitations and write-up |

Members collaborate across packages where the work overlaps (for example, the
leakage controls were reviewed jointly by the data and modeling leads), and all
four share the total estimated effort. The final report will record the specific
per-member split for the second-half tasks.

## References

Branco, R. R., Rubesam, A., & Zevallos, M. (2024). Forecasting realized
volatility: Does anything beat linear models? *Journal of Empirical Finance*, 78,
101524.

Corsi, F. (2009). A simple approximate long-memory model of realized volatility.
*Journal of Financial Econometrics*, 7(2), 174–196.

Moreno-Pino, F., & Zohren, S. (2024). DeepVol: Volatility forecasting from
high-frequency data with dilated causal convolutions. *Quantitative Finance*,
24(8), 1105–1127.

Zhang, C., Zhang, Y., Cucuringu, M., & Qian, Z. (2024). Volatility forecasting
with machine learning and intraday commonality. *Journal of Financial
Econometrics*, 22(2), 492–530.
