# Stage 3: Risk-Parity Portfolio — Complete Documentation

> **What Stage 3 does**: Takes the 23-stock universe and 16 years of data from Stages 1-2, and builds a **risk-parity portfolio** — one where every stock contributes equal risk, not equal dollars. Then simulates it over the full history, compares it against simpler alternatives, and answers the question: "Does this sophisticated approach actually earn its complexity?"

---

## Table of Contents

1. [What Stage 3 Is and Why It Exists](#1-what-stage-3-is-and-why-it-exists)
2. [Architecture Overview](#2-architecture-overview)
3. [Part A: The Optimizer (risk_parity.py)](#3-part-a-the-optimizer)
4. [Part B: The Rebalancer (rebalancer.py)](#4-part-b-the-rebalancer)
5. [Part C: The Simulator (portfolio_sim.py)](#5-part-c-the-simulator)
6. [Part D: The Comparison (compare.py)](#6-part-d-the-comparison)
7. [Key Results](#7-key-results)
8. [File Reference](#8-file-reference)
9. [Decision Log](#9-decision-log)
10. [Rectifications and Bug Fixes](#10-rectifications-and-bug-fixes)
11. [Concept Glossary](#11-concept-glossary)

---

## 1. What Stage 3 Is and Why It Exists

### The problem with equal-weight portfolios

In Stage 2, we used equal-weight allocation — give each of the 23 stocks the same dollar amount. This seems fair, but it ignores a critical fact: **not all stocks are equally risky**.

A volatile bank stock (AXISBANK, ~35% annualized vol) contributes far more portfolio risk than a defensive consumer stock (HINDUNILVR, ~22% vol). Equal dollars ≠ equal risk. Our data showed that in an equal-weight portfolio, the top risk contributor (AXISBANK) was responsible for 7.28% of total risk while the bottom (CIPLA) contributed only 2.59% — nearly a 3:1 imbalance, when the "equal" target is 4.35% each.

### What risk parity does about it

Risk parity adjusts weights so that every stock contributes the **same fraction of total portfolio risk**. Stocks that are volatile or highly correlated with the rest get less weight. Stocks that are defensive or diversifying get more. The target: each of the 23 stocks contributes exactly 1/23 = 4.35% of total risk.

### What Stage 3 builds, step by step

We built it in four parts, each dependent on the previous:

```
Part A: OPTIMIZER              Part B: REBALANCER           Part C: SIMULATOR           Part D: COMPARISON
=================              ==================           ================            =================
Given a covariance             Given target weights         Chain A + B across          Run three strategies
matrix, find weights    --->   and current holdings,  --->  187 monthly dates    --->   side by side and
where every stock              decide which trades          with real prices             isolate what each
contributes equal risk         to actually make             to get an equity curve       component adds
```

The covariance estimation and risk decomposition (MCR/CTR) were already built in Stage 3's first phase and are documented in the Stage 1-2 documentation. This document covers Parts A through D.

---

## 2. Architecture Overview

### How everything connects

```
data_access.py (Stage 1)
       |
       v
covariance.py ---------> risk_engine.py
  Ledoit-Wolf              Rolling monthly Σ + MCR/CTR decomposition
  shrinkage                        |
                                   v
                          risk_parity.py  (Part A)
                            Optimizer: Σ --> weights
                                   |
                                   v
                          rebalancer.py   (Part B)
                            Drifted weights vs targets --> trade list
                                   |
                                   v
                          portfolio_sim.py (Part C)
                            Walk forward day by day, fill at next-day open
                                   |
                                   v
                          compare.py      (Part D)
                            Risk parity vs EW rebalanced vs buy-and-hold
```

### Key architectural principle: same pipeline for all strategies

Parts A and B were designed so the simulator (Part C) could use the **exact same code path** for both risk-parity and equal-weight strategies. The only difference is what produces the target weights — the optimizer (Part A) or a simple `1/N` constant. The rebalancer, cost model, fill timing, and equity-curve tracking are shared. This eliminates the possibility of accidentally comparing two strategies that differ in some hidden assumption.

### Dependencies from earlier stages

Stage 3 depends on:
- **data_access.py** — the single gateway to the price database (`get_close_matrix`, `get_prices`, `get_calendar`)
- **covariance.py** — the Ledoit-Wolf estimator (documented in Stage 1-2 docs)
- **risk_engine.py** — the rolling covariance wrapper + MCR/CTR decomposition (documented in Stage 1-2 docs)

Stage 3 does NOT touch:
- The database directly (always goes through `data_access.py`)
- Stage 2's strategy code, portfolio logic, or engine (completely independent)

---

## 3. Part A: The Optimizer

**File**: `stage3/risk_parity.py`

### What it does

Given one covariance matrix (Σ), find 23 portfolio weights where every stock contributes equal risk.

### How it works, step by step

**Step 1: Start with a smart initial guess**

Rather than starting from equal weights (which could be far from optimal), we start with **inverse-volatility weights**: give less to volatile stocks, more to calm ones. This is already close to the risk-parity solution and helps the solver converge faster.

```
weight_i = (1 / vol_i) / sum(1 / vol_j for all j)
```

**Step 2: Define what "equal risk" means mathematically**

The objective function measures how far each stock's risk share is from the target (1/23):

```
Objective = sum( (risk_share_i - 1/23)^2  for all i )
```

Where `risk_share_i = (w_i * (Σw)_i) / (w'Σw)` — this is each stock's fraction of total portfolio variance. The solver minimizes this sum of squared deviations.

**Step 3: Run the solver**

We use scipy's SLSQP optimizer with two constraints:
- All weights must sum to 1 (fully invested)
- Each weight between 0 and 10% (no shorting, no single stock > 10%)

The 10% cap is a deliberate choice — without it, a very low-vol stock could get an unreasonably large weight.

**Step 4: Verify the result**

After the solver returns, we don't just trust it. We independently compute risk shares using the existing `decompose_risk()` function (which was already verified to machine-epsilon precision). If the maximum gap between any stock's risk share and 1/23 exceeds 0.5 percentage points, we declare failure.

**Step 5: Handle failure**

If the solver doesn't converge, we fall back to the previous month's weights (or to the inverse-vol initial guess on the very first month). We print a loud warning but don't crash — the simulation should continue. In practice, across all 187 rebalance dates: **0 failures**.

### Configuration choices

| Parameter | Value | Rationale |
|---|---|---|
| Max weight per stock | 10% | Prevents over-concentration in low-vol stocks |
| Solver method | SLSQP | Handles bounds + equality constraints natively |
| Convergence tolerance | 1e-8 | Tight enough for financial precision |
| Max iterations | 1,000 | Never needed — solver typically converges in ~30 iterations |
| Risk share tolerance | 0.5pp | Maximum acceptable gap between any stock's risk share and 1/23 |

### What the output looks like

The `RiskParityResult` dataclass returns not just weights, but:
- Risk shares per stock (to verify equality)
- The maximum risk-share gap (to quantify how close to perfect)
- Whether the solver converged
- Portfolio volatility at the optimal weights
- Number of iterations used

### Verification results

Tested on the 2020-06-01 covariance matrix (COVID period, high correlations):
- All 23 risk shares within 0.1pp of 4.35% target
- Maximum gap: 0.07pp (well within 0.5pp tolerance)
- Solver converged in ~25 iterations
- Result independently verified against hand-computed risk decomposition

---

## 4. Part B: The Rebalancer

**File**: `stage3/rebalancer.py`

### What it does

Compares drifted-actual portfolio weights against the optimizer's fresh targets, and decides stock-by-stock which trades to actually make.

### Why this is a separate component

Between rebalance dates, stock prices move and portfolio weights drift. On a rebalance date, the optimizer produces new target weights, but the portfolio's actual weights have drifted from last month's targets. The rebalancer compares "where you are" to "where you should be" and generates the trade list.

### How it works, step by step

**Step 1: Compute current (drifted) weights**

From the current share holdings and today's prices, calculate what each stock's weight actually is right now.

**Step 2: Compute the gap per stock**

`gap_i = target_weight_i - current_weight_i`

Positive gap = underweight, need to buy. Negative gap = overweight, need to sell.

**Step 3: Apply the threshold filter**

If `|gap_i|` is smaller than the threshold, skip that stock — the trade isn't worth the cost. This is the key parameter we tested in Part C.

**Step 4: Convert to whole shares**

Convert the weight gap to a rupee amount (`gap * portfolio_value`), then divide by the stock price. Floor to whole shares (NSE doesn't allow fractional shares). If rounding eats the entire trade (e.g., a Rs 200 gap on a Rs 5,000 stock), log the stock as "skipped due to rounding."

**Step 5: Apply trading costs**

15 bps per leg (matching Stage 2's cost model exactly). Cost = notional traded × 0.0015.

### The threshold parameter

The threshold is **not hardcoded** — it's a parameter we sweep in Part C. The question it answers: "How far must a stock drift from target before the cost of trading back is worth it?"

| Threshold | Meaning |
|---|---|
| 0.00% | Trade everything every month, no matter how small the gap |
| 0.25% | Skip trades where the gap is less than 0.25 percentage points |
| 0.50% | Skip trades where the gap is less than 0.5 percentage points |
| 1.00% | Only trade when a stock has drifted more than 1 full percentage point |

### What the output looks like

`RebalanceResult` includes:
- List of trades (ticker, direction, shares, price, cost, weight gap)
- List of skipped stocks (with the reason: below threshold or eaten by rounding)
- Total buy/sell notional, total cost, turnover
- Portfolio value before and after

---

## 5. Part C: The Simulator

**File**: `stage3/portfolio_sim.py`

### What it does

Chains the optimizer and rebalancer across all 187 monthly rebalance dates, with real prices, producing a daily equity curve for the full 2011–2026 period.

### How it works, step by step

**Step 1: Set up starting capital**

Rs 10,00,000 (₹10L) — same as Stage 2, so the two stages are directly comparable.

**Step 2: Build the initial portfolio**

On the first rebalance date (Feb 1, 2011), buy all 23 stocks at the optimizer's target weights. This first purchase **always uses threshold=0** regardless of the run's configured threshold — you need a full initial allocation. Entry cost is paid once, at 15bps per leg, same as Stage 2.

**Step 3: Walk forward day by day**

For each of the 3,839 trading days:

- **Non-rebalance days**: Do nothing. Just mark-to-market (update portfolio value from today's close prices). Weights drift naturally as prices move.
- **Rebalance days**: Run the full monthly cycle (next step).

**Step 4: The monthly rebalance cycle**

On a rebalance date, in this exact order:
1. Get today's Σ from the risk engine (pre-computed)
2. Run the optimizer to get fresh target weights
3. Compute actual (drifted) weights from current holdings
4. Hand both to the rebalancer, which decides trades
5. **Queue the trades for execution at tomorrow's open price** — this is the no-look-ahead rule, same as Stage 2. The decision uses today's close prices, but the fill happens at the next available price.
6. On the next trading day, apply the trades at open prices and deduct costs

**Step 5: Track daily risk shares**

We compute `decompose_risk()` on the actual portfolio weights **every single day**, not just on rebalance dates. This tells us how far the portfolio drifts from equal risk contribution between rebalances — a key question for evaluating whether looser thresholds let risk parity "decay" too much.

**Step 6: Run the threshold sweep**

We run the entire simulation four times, once for each threshold (0%, 0.25%, 0.5%, 1.0%), to find the optimal cost-vs-drift tradeoff.

### The fill timing rule (non-negotiable)

This is the most important correctness property in the entire simulation:

```
Day T (close):  Get Σ, run optimizer, decide trades    <-- uses information available at close
Day T+1 (open): Execute trades at open price            <-- actual fill price
```

The decision is based on information available through today's close. The fill must happen at the **next available price**, since you couldn't have traded at a price before knowing the decision. This was verified by spot-checking 3 rebalance dates per threshold run — decision and fill dates are always consecutive trading days.

### Verification results

| Check | Result |
|---|---|
| Cash + holdings reconciliation | **PASS** — all 4 thresholds × 3,839 days = 15,356 portfolio-days, zero errors |
| No-look-ahead spot check | **PASS** — 12 spot checks (3 per threshold), all correct |
| Entry cost applied once | **PASS** — first rebalance uses threshold=0 (deliberate override), subsequent use configured threshold |
| Optimizer convergence | **PASS** — 0 failures across all 187 dates × 4 runs |

### Threshold sweep results

| Threshold | Total Return | CAGR | Sharpe | Max DD | Cumulative Cost | Cumulative Turnover |
|---|---|---|---|---|---|---|
| 0.00% | 1,025% | 16.82% | 1.200 | -31.87% | Rs 80,257 | 11.81x |
| 0.25% | 1,016% | 16.76% | 1.196 | -31.88% | Rs 64,469 | 9.77x |
| 0.50% | 994% | 16.61% | 1.186 | -32.35% | Rs 46,352 | 7.42x |
| **1.00%** | **997%** | **16.63%** | **1.190** | **-31.77%** | **Rs 32,069** | **5.16x** |

### Why we chose the 1% threshold

The 1% threshold cuts costs by 60% (Rs 32K vs Rs 80K) while giving up essentially nothing in performance (CAGR: 16.63% vs 16.82%, Sharpe: 1.190 vs 1.200). It also has the best max drawdown (-31.77%). There's no real cost-vs-drift tradeoff here — the looser threshold simply wins.

The max drawdown was not monotonic across thresholds (0.5% had the worst at -32.35%). We investigated and confirmed this is pure path dependence — all four thresholds hit their max drawdown during the COVID crash (peak Jan 17, 2020 → trough Mar 23, 2020), but with slightly different weight vectors entering the crash due to different trade histories. The 58bp difference is noise, not signal.

### Risk-share drift analysis

At the chosen 1% threshold, how much does risk parity "decay" between rebalances?

| Metric | Value |
|---|---|
| Mean absolute deviation from 1/23 | 0.50 percentage points |
| 95th percentile deviation | 1.18 percentage points |
| Max deviation ever observed | 3.96 percentage points |

Verdict: even with the loosest threshold, risk parity stays close to equal risk contribution. The portfolio doesn't meaningfully "decay" between monthly rebalances.

---

## 6. Part D: The Comparison

**File**: `stage3/compare.py`

### What it does

Runs three strategies over the **exact same window** (Feb 2011 – Aug 2026), with the same cost model, same universe, same starting capital, and isolates what each component adds.

### The three strategies

| Strategy | What it does | What it tests |
|---|---|---|
| **Risk Parity** | Monthly rebalance to risk-parity weights, 1% threshold | The full system |
| **EW Rebalanced** | Monthly rebalance to 1/23 weights, 1% threshold | "Does just rebalancing matter?" |
| **Buy & Hold** | Buy equal-weight once, never rebalance | The passive baseline |

### How the equal-weight control was built

The equal-weight-rebalanced strategy reuses **the exact same simulator code** as risk parity. The only change: instead of calling the optimizer (Part A), it uses a fixed target of 1/23 for every stock, every month. Same rebalancer, same cost model, same no-look-ahead fill rule, same 1% threshold.

This was achieved by adding a `weight_source` parameter to `run_simulation()`:
- `weight_source="risk_parity"` → calls the optimizer
- `weight_source="equal_weight"` → uses 1/N

No code duplication, no risk of hidden differences between the two runs.

### How the buy-and-hold baseline was built

The buy-and-hold strategy is a separate, simpler function: buy all 23 stocks equally at close on the first day, pay the one-time entry cost (15bps per leg, same as the others), then hold forever. No rebalancing, no threshold, no optimizer.

### Verification

| Check | Result |
|---|---|
| Dates match | **PASS** — all three cover the same 3,839 trading days |
| Cash + holdings reconciliation | **PASS** — all three strategies |
| Entry cost consistency | **PASS** — Rs 1,493–1,496 (tiny rounding diff) |
| EW targets genuinely flat | **PASS** — spot-checked 5 rebalance dates, all weights = 1/23 |
| Effect decomposition sums correctly | **PASS** — CAGR and Sharpe reconstruct exactly |

### Three-way comparison results

| Strategy | Total Return | CAGR | Sharpe | Sortino | Max DD | Cumulative Cost |
|---|---|---|---|---|---|---|
| Risk Parity | 997% | 16.63% | **1.190** | **1.608** | **-31.77%** | Rs 32,069 |
| EW Rebalanced | 877% | 15.76% | 1.094 | 1.461 | -34.06% | Rs 14,765 |
| Buy & Hold | **1,021%** | **16.80%** | 1.073 | 1.464 | -34.06% | Rs 1,493 |

### The two isolated effects — the actual point of Stage 3

This is the key analysis. We decomposed the total risk-parity-vs-buy-and-hold gap into two separate effects:

**Effect 1: Rebalancing alone** (EW Rebalanced minus Buy & Hold)

| Metric | Effect |
|---|---|
| CAGR | **-1.04%** (rebalancing destroys return) |
| Sharpe | +0.021 (tiny improvement) |
| Sortino | -0.003 (essentially zero) |
| Max DD | +0.00% (no improvement) |

**Interpretation**: Monthly equal-weight rebalancing alone is slightly **value-destructive**. The -1.04% CAGR drag is entirely trading costs eating into natural compounding that buy-and-hold captures for free. The Sharpe improvement is real but negligible.

**Effect 2: Risk weighting** (Risk Parity minus EW Rebalanced)

| Metric | Effect |
|---|---|
| CAGR | **+0.87%** (risk weighting adds return) |
| Sharpe | **+0.097** (meaningful improvement) |
| Sortino | **+0.147** (even larger — reduces downside specifically) |
| Max DD | **+2.29%** (shallower drawdown) |

**Interpretation**: The risk-parity optimizer **earns its complexity**. It adds 87bps of CAGR, a meaningful Sharpe improvement, and 2.3 percentage points less drawdown — all on top of what simple rebalancing provides. The Sortino improvement being larger than the Sharpe improvement tells us risk parity specifically reduces **downside** volatility, which is exactly what an equal-risk-contribution portfolio should do.

**Sanity check**: The two effects sum exactly:
- CAGR: rebal (-1.04%) + risk-wt (+0.87%) = -0.17%, actual total = -0.17% ✓
- Sharpe: rebal (+0.021) + risk-wt (+0.097) = +0.118, actual total = +0.118 ✓

### The max drawdown coincidence

EW Rebalanced and Buy & Hold both show -34.06% max drawdown. We investigated whether this was a data-handling slip (e.g., accidentally comparing the same equity curve). Finding: the drawdowns are **not identical at full precision** — they differ by 0.0048 percentage points. They're not even measuring from the same peak (BH peaked Aug 2018, EW peaked Jan 2020). The weight vectors on the pre-crash peak date are substantially different (correlation 0.48, max absolute weight difference 16.7pp for Eicher Motors). The near-identical drawdown at 2 decimal places is genuinely coincidental — COVID hit everything broadly.

### Risk-share drift comparison

This confirms the optimizer is actually doing something, not just coincidentally producing similar returns:

| Strategy | Mean Deviation from 1/23 | 95th Percentile | Max Deviation |
|---|---|---|---|
| Risk Parity | 0.50% | 1.18% | 3.96% |
| EW Rebalanced | 1.26% | 2.94% | 7.47% |
| Buy & Hold | 3.28% | 4.08% | **63.96%** |

Equal-weight's risk imbalance is **2.5x** that of risk parity. Buy-and-hold eventually lets a single stock dominate the portfolio's risk profile entirely.

### The 64% risk-share concentration: Eicher Motors

The extreme 63.96% max deviation in buy-and-hold was traced to a specific stock and date:

- **Stock**: Eicher Motors (EICHERMOT.NS)
- **Date**: August 21, 2017
- **Dollar weight**: 36.3% (grew from ~4.3% at entry through price appreciation alone)
- **Risk share**: 68.3% — nearly two-thirds of total portfolio risk from one stock
- **Risk/weight ratio**: 1.88x — Eicher wasn't just big, it was volatile-big

Eicher's ~8x price appreciation over 6 years (2011–2017) turned it into a position that dominated both the portfolio's capital allocation and its risk profile. A single stock controlled 68% of total risk while the target is 4.35% — a 16x concentration. This is the exact problem risk parity exists to prevent.

---

## 7. Key Results

### The honest summary

**Buy-and-hold wins on raw return** (16.80% CAGR vs 16.63%). You can't beat doing nothing when the market goes up 10x over 16 years, because every trade has a cost and buy-and-hold pays only once.

**But risk parity delivers the best risk-adjusted ride**:
- Best Sharpe (1.190 vs 1.073 for buy-and-hold)
- Best Sortino (1.608 vs 1.464)
- Shallowest drawdown (-31.77% vs -34.06%)

**And the improvement comes from the risk-parity math itself**, not just from rebalancing. Rebalancing alone is slightly value-destructive; it's the risk-weighted allocation that earns the improvement.

### What this means in practical terms

If you invest Rs 10 lakh and hold for 15 years:
- **Buy-and-hold**: Rs 1.12 crore, but you'd sit through a -34% drawdown (Rs 15.7L peak-to-trough loss)
- **Risk parity**: Rs 1.10 crore (Rs 2.4L less), but your worst drawdown is -31.8% (Rs 13.2L) — you save Rs 2.5L of peak-to-trough pain

The question is whether the smoother ride is worth Rs 2.4L of raw return. There's no objectively correct answer — it depends on the investor.

---

## 8. File Reference

### stage3/ directory — Stage 3 (Risk-Parity Portfolio)

| File | Part | Purpose |
|---|---|---|
| `covariance.py` | (prereq) | Point-in-time Ledoit-Wolf covariance estimator |
| `risk_engine.py` | (prereq) | Rolling monthly wrapper + MCR/CTR risk decomposition |
| `risk_parity.py` | Part A | Risk-parity optimizer: Σ → weights where every stock contributes equal risk |
| `rebalancer.py` | Part B | Compares drifted weights vs targets, generates trade list with threshold filter |
| `portfolio_sim.py` | Part C | Full 16-year walk-forward simulator with daily equity curve and threshold sweep |
| `compare.py` | Part D | Three-way comparison: risk parity vs EW rebalanced vs buy-and-hold |
| `verify_risk_parity.py` | verify | Tests optimizer convergence, fallback behavior, risk-share equality |
| `verify_rebalancer.py` | verify | Tests threshold filtering, rounding, cost calculation, edge cases |
| `verify_risk_engine.py` | verify | Tests covariance matrix properties, rolling wrapper, MCR/CTR identity |
| `verify_independent.py` | verify | Hand-verification of 4 specific claims from automated checks |
| `verify_rp_independent.py` | verify | Independent cross-check of optimizer results |

### Output files (root directory)

| File | Contents |
|---|---|
| `stage3_equity_curve_t0.0000.csv` | Daily equity curve, 0% threshold |
| `stage3_equity_curve_t0.0025.csv` | Daily equity curve, 0.25% threshold |
| `stage3_equity_curve_t0.0050.csv` | Daily equity curve, 0.5% threshold |
| `stage3_equity_curve_t0.0100.csv` | Daily equity curve, 1% threshold |
| `stage3_threshold_comparison.csv` | Metrics table for all 4 thresholds |
| `stage3_risk_share_drift.csv` | Daily risk shares (23 columns) at 0.5% threshold |
| `stage3_compare_risk_parity.csv` | Equity curve, risk parity 1% threshold (Part D) |
| `stage3_compare_ew_rebalanced.csv` | Equity curve, equal-weight rebalanced (Part D) |
| `stage3_compare_buy_and_hold.csv` | Equity curve, buy-and-hold (Part D) |
| `stage3_comparison_metrics.csv` | Three-way metrics comparison (Part D) |
| `stage3_ew_risk_share_drift.csv` | Daily risk shares for EW rebalanced portfolio |

---

## 9. Decision Log

Every deliberate design choice in Stage 3, with reasoning.

### Part A decisions

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 30 | Optimizer objective | Minimize sum of squared deviations of risk shares from 1/N | Standard risk-parity formulation. Works in variance space (no sqrt), gives smoother gradients for the solver. |
| 31 | Solver method | SLSQP (Sequential Least Squares Programming) | Natively handles bounds + equality constraints. No need to reformulate as unconstrained. |
| 32 | Initial guess | Inverse-volatility weights | Much closer to the solution than equal weights. Helps solver converge in fewer iterations. |
| 33 | Max weight cap | 10% per stock | Prevents over-concentration. Without it, a very low-vol stock could get an unreasonably large weight. |
| 34 | Failure handling | Fall back to previous month's weights, log loudly | Don't crash the simulation. A "no change" month is a valid outcome. In practice: 0 failures across all runs. |
| 35 | Convergence check | Independent risk-share verification, not just solver's own success flag | The solver might claim success but still leave risk shares unequal. We check the actual result, not just the solver's report. |

### Part B decisions

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 36 | Cost model | 15bps per leg, matching Stage 2 | Same assumptions across the project. Changes only in config.py if we ever update. |
| 37 | Threshold default | 0.5% (but swept from 0% to 1%) | The default is just a starting point. The sweep in Part C is what actually determines the optimal threshold. |
| 38 | Share rounding | Floor to whole shares | NSE doesn't allow fractional shares. Floor (not round) is conservative — we slightly under-trade rather than over-trade. |
| 39 | Rounding → skip | If rounding eats the entire trade, log as "skipped due to rounding" | Distinguishes "gap too small to trade" from "gap was tradeable but the stock price was too high for even one share." |

### Part C decisions

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 40 | Fill timing | Signal at close, fill at next day's open | Same no-look-ahead rule as Stage 2. Non-negotiable. Verified by spot-checking 12 rebalance dates. |
| 41 | First rebalance override | threshold=0 regardless of run's setting | You need a full initial allocation. Can't start with partial holdings because the threshold filtered out some stocks. |
| 42 | Daily risk-share logging | Compute decompose_risk() every day, not just rebalance dates | Shows how risk parity decays between rebalances. Cheap to compute (matrix multiply, not optimization). Worth the log volume. |
| 43 | Starting capital | Rs 10,00,000 | Matches Stage 2 for direct comparison. |
| 44 | Chosen threshold | 1.0% | Cuts costs by 60%, gives up <20bps of CAGR, has best max drawdown. Clear winner. |

### Part D decisions

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 45 | Comparison window | Feb 2011 – Aug 2026 (same as risk parity's first usable date) | All three strategies must cover the exact same dates. Verified to the day. |
| 46 | EW rebalanced threshold | 1.0% (same as risk parity) | Isolates the weighting effect only. Using a different threshold would confound the comparison. |
| 47 | Code reuse for EW control | Same run_simulation() with weight_source="equal_weight" | Zero code duplication, zero chance of hidden differences. |
| 48 | Buy-and-hold entry cost | Same 15bps per leg as the others | Fair comparison. The benchmark isn't artificially advantaged by free entry. |
| 49 | Effect decomposition | Compute both rebalancing effect and risk-weighting effect separately | The whole point — "does the optimizer earn its complexity, or is rebalancing alone enough?" |

---

## 10. Rectifications and Bug Fixes

Issues we caught and fixed during development.

### Unicode encoding error (Part C)

**Problem**: The simulator used ₹ (Unicode rupee symbol, U+20B9) in print statements. Windows PowerShell uses cp1252 encoding, which can't render this character. The simulation crashed with `UnicodeEncodeError` on the first rebalance log line.

**Fix**: Replaced all `₹` with `Rs` in print statements. Pure display issue, no impact on calculations.

### Ticker resolution bug (Part C)

**Problem**: The initial code to resolve tickers from the covariance engine used a Python short-circuit expression (`sorted(engine.rebalance_dates[0] and engine.get_estimate(...).tickers)`) that could evaluate incorrectly due to Timestamp truthiness.

**Fix**: Replaced with a clean two-line lookup: get the estimate first, then sort its tickers. No impact on results (the bug wouldn't have produced wrong answers in practice, but the code was fragile).

### Missing dependencies

**Problem**: The project's virtual environment (`c:\Users\Pratham\venv`) didn't have scipy, pandas, or scikit-learn installed when we started Stage 3.

**Fix**: Installed via pip: `scipy==1.18.1`, `numpy==2.5.3`, `pandas==3.0.6`, `scikit-learn==1.9.1`.

---

## 11. Concept Glossary

New concepts introduced in Stage 3 (see Stage 1-2 documentation for earlier concepts).

| Concept | Plain English |
|---|---|
| **Risk parity** | An allocation where every stock contributes equal risk. Not equal dollars — equal *risk contribution*. Volatile stocks get less weight, defensive stocks get more. |
| **Risk share** | Each stock's fraction of total portfolio risk. In a 23-stock risk-parity portfolio, the target is 1/23 = 4.35% each. |
| **Objective function** | The math expression the optimizer tries to minimize. Ours: sum of squared deviations of risk shares from 1/N. When it reaches zero, all risk shares are exactly equal. |
| **SLSQP** | Sequential Least Squares Programming. A numerical optimization method that handles constraints (like "weights must sum to 1" and "no weight above 10%"). |
| **Threshold (rebalance)** | Minimum weight drift required before a trade is triggered. A 1% threshold means: don't trade a stock unless its weight has drifted more than 1 percentage point from target. Reduces trading costs at the expense of precision. |
| **Turnover** | Total notional amount traded divided by portfolio value. Higher turnover = more trading = more costs. |
| **Drifted weights** | The weights a portfolio naturally evolves to between rebalances, as stock prices change. If stock A goes up 10% and stock B goes down 10%, A's weight drifts up and B's drifts down. |
| **Sortino ratio** | Like Sharpe, but only penalizes *downside* volatility. Ignores upside moves. Higher = better. More relevant than Sharpe for strategies designed to reduce drawdowns. |
| **Path dependence** | When the final outcome depends on the exact sequence of events, not just the start and end points. Two portfolios with identical long-run returns can have different max drawdowns because of *when* their worst days happened relative to their peak. |
| **Rebalancing drag** | The cost of trading back to target weights every month. In a strongly trending market, rebalancing sells winners and buys losers — dragging returns below buy-and-hold. Our data showed -1.04% CAGR drag from rebalancing alone. |
| **Concentration risk** | The danger of one stock dominating your portfolio's risk. Our buy-and-hold portfolio let Eicher Motors reach 68% of total risk — nearly two-thirds of risk from a single stock. |
| **Risk/weight ratio** | A stock's risk share divided by its dollar weight. If a stock is 5% of the portfolio but contributes 10% of risk, its risk/weight ratio is 2.0x. Above 1.0 = disproportionately risky for its size. |
