# Quantitative Portfolio System — Complete Project Documentation

> **What this project is**: A quantitative portfolio backtesting and risk analysis system for 23 large-cap Indian (NSE) stocks, covering 16 years of daily data (Jan 2010 - Aug 2026). It downloads, cleans, and stores historical price data; runs two trading strategies simultaneously; and analyzes portfolio risk using modern covariance estimation.

---

## Table of Contents

1. [Project Overview and Architecture](#1-project-overview-and-architecture)
2. [Stage 1: Data Pipeline](#2-stage-1-data-pipeline)
3. [Stage 2: Backtester](#3-stage-2-backtester)
4. [Stage 3: Risk Engine](#4-stage-3-risk-engine)
5. [Key Results](#5-key-results)
6. [File Reference (every file explained)](#6-file-reference)
7. [Decision Log (every design choice)](#7-decision-log)
8. [Concept Glossary](#8-concept-glossary)

---

## 1. Project Overview and Architecture

### What problem does this solve?

You have 23 large-cap Indian stocks. You want to know:
- **Stage 2**: If I trade these stocks using two simple strategies (moving average crossover + RSI mean reversion), how would I have performed over 16 years?
- **Stage 3**: How much risk does each stock contribute to my portfolio, and how do the stocks move together?

### The three stages

```
Stage 1: DATA PIPELINE        Stage 2: BACKTESTER           Stage 3: RISK ENGINE
========================      =======================       ====================
Download raw prices     -->   Generate signals        -->   Estimate covariance
Clean & validate              Size positions                Decompose risk
Store in SQLite DB            Simulate fills + costs        (feeds into optimizer)
                              Measure performance
```

### Architecture principle: decoupling

Every component knows ONLY what it needs to:
- Strategy doesn't know Portfolio exists (it just says BUY/SELL/HOLD)
- Portfolio doesn't know ExecutionHandler's cost model (it just sizes orders)
- ExecutionHandler doesn't know why a trade happened (it just fills orders)
- Engine is the ONLY component that connects everything together

This means you can swap any component without touching the others.

### The 23 stocks

All from NSE (National Stock Exchange of India), across 7 sectors:

| Sector | Tickers |
|---|---|
| **Auto** | EICHERMOT.NS, M&M.NS, MARUTI.NS |
| **Banking** | AXISBANK.NS, HDFCBANK.NS, ICICIBANK.NS, KOTAKBANK.NS, SBIN.NS |
| **Energy** | NTPC.NS, ONGC.NS, RELIANCE.NS |
| **FMCG** | HINDUNILVR.NS, ITC.NS, NESTLEIND.NS |
| **IT** | HCLTECH.NS, INFY.NS, TCS.NS, WIPRO.NS |
| **Infrastructure** | LT.NS, ULTRACEMCO.NS |
| **Pharma** | CIPLA.NS, DRREDDY.NS, SUNPHARMA.NS |

### Database

SQLite database at `db/quant.db` with 5 tables:

| Table | Purpose | Key columns |
|---|---|---|
| `tickers` | Master list of 23 stocks | symbol, sector, exchange |
| `daily_prices` | OHLCV data, one row per (ticker, date) | open, high, low, close, volume, data_quality |
| `trading_calendar` | Consensus trading days across all tickers | trade_date |
| `corporate_actions` | Splits, bonuses, dividends | action_type, ratio_or_amount |
| `data_quality_log` | Known data issues and their resolutions | issue_type, description, resolution |

The `data_quality` column on every price row is one of:
- `clean` -- verified good data
- `forward_filled` -- price was missing, filled with previous day's close
- `vol_missing` -- price is fine but volume data is missing

---

## 2. Stage 1: Data Pipeline

### Purpose

Get 16 years of clean, validated daily price data for 23 stocks into a SQLite database. The key principle: **fail loudly, not silently**. If data is bad, raise an error -- never silently fix it and hope for the best.

### How data flows

```
Yahoo Finance API (yfinance)
        |
        v
download_nifty_data.py  -->  Raw CSV/pickle files in data/
        |
        v
load_to_db.py           -->  Insert into db/quant.db
        |
        v
reconcile_calendar.py   -->  Build trading_calendar (consensus dates)
        |
        v
fix_stale_rows.py       -->  Tag forward-filled rows as data_quality='forward_filled'
        |
        v
gap_check.py            -->  Check for remaining gaps
        |
        v
final_validation.py     -->  Full integrity check (the "final exam")
```

### Key concepts in Stage 1

**Adjusted prices**: All prices are "adjusted" -- meaning stock splits and bonuses are retroactively applied. If TCS split 1:10, all historical prices are divided by 10 so the chart looks continuous. This is done by yfinance automatically (`auto_adjust=True`). Without adjustment, a 1:10 split would look like the stock crashed 90% overnight.

**Trading calendar reconciliation**: Not all stocks trade on the same days (different listing dates, different suspensions). The `trading_calendar` table contains only dates where ALL 23 stocks have data -- a "rectangular matrix" with no gaps. This is important because the backtester and risk engine need to iterate over dates that work for all stocks simultaneously.

**Data quality tagging**: Rather than silently filling gaps, every row is tagged with its quality. Only 28 out of ~94,500 rows are non-clean (0.03%), all tagged as `stale_row` (forward-filled). This is immaterial but documented honestly.

### Scripts in Stage 1

| Script | What it does |
|---|---|
| `download_nifty_data.py` | Downloads OHLCV data from Yahoo Finance for all 23 tickers |
| `load_to_db.py` | Inserts downloaded data into `db/quant.db`, applies schema |
| `reconcile_calendar.py` | Builds the `trading_calendar` table from the intersection of all tickers' trading dates |
| `fix_stale_rows.py` | Detects "stale" rows (identical OHLC to previous day) and tags them as forward-filled |
| `gap_check.py` | Scans for missing dates, gaps in the data |
| `final_validation.py` | Comprehensive integrity check: all tickers present, all dates covered, no NaNs, high >= low, etc. |
| `debug_tickers.py` | Ad-hoc debugging script used during development |
| `test_db.py` | Tests that the database schema and queries work correctly |
| `generate_summary.py` | Generates summary statistics for the dataset |

### Data Access Layer

`data_access.py` is the ONLY interface between the database and all downstream code. No other file ever touches `quant.db` directly. It provides:

| Function | Returns |
|---|---|
| `get_prices(symbol, start, end)` | OHLCV DataFrame for one ticker |
| `get_ticker_info()` | Metadata for all 23 tickers |
| `get_calendar(start, end)` | Trading calendar dates |
| `get_close_matrix(start, end)` | Dates x tickers close-price matrix (23 columns) |
| `get_quality_issues()` | Data quality log entries |

---

## 3. Stage 2: Backtester

### Purpose

Simulate running two trading strategies simultaneously on the 23 stocks over 16 years, with realistic transaction costs, and measure performance against a benchmark.

### The two strategies

#### 1. Moving Average Crossover (MA_CROSSOVER)

**Concept**: Two moving averages of different lengths. When the short-term average crosses above the long-term average, the stock's trend has turned bullish (BUY). When it crosses below, the trend is bearish (SELL).

**Parameters**: Short window = 20 days, Long window = 50 days

**Signal logic**:
- BUY: SMA(20) > SMA(50) -- short-term trend is above long-term
- SELL: SMA(20) < SMA(50) -- short-term trend has dropped below long-term
- HOLD: Not enough data to compute both averages

**Score** (conviction measure): `|SMA_short - SMA_long| / SMA_long` -- how far apart the two averages are, as a percentage. A wider gap means stronger trend conviction.

**Why this strategy**: It's a classic trend-following approach. Simple, well-understood, captures momentum. The 20/50 day combination is standard -- not optimized for this specific dataset, which avoids overfitting.

#### 2. RSI Mean Reversion (RSI_MEANREV)

**Concept**: The Relative Strength Index measures how "overbought" or "oversold" a stock is on a 0-100 scale. When RSI drops below 30, the stock may be oversold and due for a bounce (BUY). When it rises above 70, it may be overbought and due for a pullback (SELL).

**Parameters**: Period = 14 days (Wilder's standard RSI)

**Signal logic**:
- BUY: RSI < 30 -- oversold
- SELL: RSI > 70 -- overbought
- HOLD: RSI between 30 and 70

**Score**: `30 - RSI` when oversold (how far below 30). Lower RSI = higher score = stronger conviction the stock will bounce.

**Why this strategy**: It's the opposite philosophy from MA crossover -- it buys weakness instead of strength. Running both simultaneously tests whether combining a trend-following and a mean-reversion strategy adds diversification value.

### How the backtester works (bar by bar)

On every trading day, the engine runs this loop:

```
For each date in the 16-year range:

  1. FILL yesterday's orders using today's Open price
     (this is the 1-bar signal-to-fill lag -- see below)

  2. GENERATE signals: run both strategies on every ticker
     using today's Close price and historical data

  3. RESOLVE signals into orders (Portfolio does this):
     Phase 1:   Handle exits (SELL signals, partial scale-outs)
     Phase 1.5: Conviction-based replacements (see below)
     Phase 2:   Rank and size new entries

  4. MARK TO MARKET: record today's portfolio value
```

### Key concept: Signal-to-fill lag

**The problem**: If you generate a signal using today's Close price and then fill the order AT that same Close price, you're cheating -- you're using information (the closing price) that you wouldn't know until the market closed, but pretending you could trade at that price. This is called look-ahead bias.

**The solution**: Signals generated on bar t (using the Close) are queued and filled on bar t+1 (using the Open). This is the 1-bar lag. You see today's close, decide to buy, and the earliest you could actually execute is tomorrow's open.

**Config**: `SIGNAL_TO_FILL_LAG_BARS = 1`, `FILL_PRICE_FIELD = "Open"`, `SIGNAL_PRICE_FIELD = "Close"`

### Key concept: Shared capital pool

Both strategies compete for the same money. There isn't a separate $500K for MA crossover and $500K for RSI -- it's one $1M pool. This means:
- If MA crossover wants to buy 5 stocks and RSI wants to buy 5 stocks on the same day, they compete for the same limited capital
- A position opened by one strategy is owned by that strategy -- the other strategy's opinion is ignored for that position

### Key concept: Position sizing (inverse-volatility weighting)

**The problem**: If you buy equal dollar amounts of each stock, volatile stocks dominate your portfolio's risk. Buying 100K of a stock with 40% annual volatility contributes way more risk than 100K of a stock with 15% volatility.

**The solution**: Allocate more money to low-volatility stocks and less to high-volatility stocks. Specifically, weight each stock by `1 / volatility`:

```
weight_i = (1 / vol_i) / sum(1 / vol_j for all selected stocks)
allocation_i = deployable_cash * weight_i
shares_i = floor(allocation_i / stock_price_i)  # whole shares only (NSE)
```

**Config**: `VOLATILITY_LOOKBACK = 20` trading days for measuring volatility.

### Key concept: Percentile-based cross-strategy ranking

**The problem**: MA crossover scores (percentage gap between two moving averages) and RSI scores (how far below 30) are on completely different scales. You can't directly compare them.

**The solution** (ranking.py): Convert both sets of scores into percentile ranks (0 to 1 scale). If a stock ranks in the 90th percentile of MA scores, it gets 0.9. If another ranks in the 50th percentile of RSI scores, it gets 0.5. Now they're comparable.

When both strategies produce a BUY for the same ticker, keep the higher percentile score. On exact ties, MA wins (deliberate choice, not accident).

### Key concept: Exit rules

Three ways a position can exit:

1. **Full exit (SELL signal)**: The owning strategy says SELL -- sell everything immediately
2. **Partial scale-out**: When the owning strategy's signal transitions from BUY to HOLD, sell 50% (one-time). This locks in some profit while keeping skin in the game.
3. **Conviction-based replacement** (Phase 1.5): A stronger new candidate displaces a weak existing holding (see below)

### Key concept: Conviction-based replacement (Phase 1.5)

**The problem**: Once the portfolio fills up (~20 positions), capital gets permanently locked. New candidates can't enter because `deployable cash = 0`. Three tickers (AXISBANK, HINDUNILVR, LT) were never held in 16 years despite ranking #1 thousands of times.

**Root cause investigation**: Diagnostic analysis confirmed these tickers generated frequent BUY signals and ranked highly, but `int(allocation // price)` rounded to 0 shares when the portfolio was near 95% deployment with only tiny amounts of free cash remaining.

**The solution**: Every bar, rank ALL BUY-signaled tickers (held + unheld) on the same percentile scale. If a challenger's score beats the weakest defender's score by at least 0.40 (on the 0-1 scale), the defender is fully sold and the challenger takes its place.

**Three guardrails to control churn**:

| Guardrail | Value | Purpose |
|---|---|---|
| `MIN_CONVICTION_EDGE` | 0.40 | Don't swap for a marginal improvement -- the 30bps round-trip cost needs to be worth it |
| `MIN_HOLDING_PERIOD` | 10 bars | Don't displace something you bought yesterday |
| `MAX_REPLACEMENTS_PER_DAY` | 2 | Don't swap out everything at once |

**Result**: 304 replacements over 16 years. Return dropped from 680% to 647% (extra costs > diversification benefit) but Sharpe improved slightly (0.870 to 0.882). The conclusion: **diversification gain doesn't cover the 30bps churn cost**.

**SUNPHARMA finding**: Displaced 20 times (most of any ticker), entered only 4 times (least of any ticker). It's a chronic weak-conviction holder -- gets in, then gets bumped whenever something stronger appears. This was verified from the trade log, not guessed from the final snapshot.

### Key concept: Transaction cost model

Every trade incurs a flat 30 basis points (0.30%) round-trip cost, split 15bps per leg. This blends together:
- Brokerage fees
- Securities Transaction Tax (STT)
- Stamp duty
- Estimated slippage (market impact)

This is applied by ExecutionHandler at fill time, not by Portfolio at sizing time.

### Key concept: Benchmark comparison

**The problem**: If your strategy returned 680% over 16 years, is that good? You can't tell without a reference point.

**The solution** (benchmark.py): Build an equal-weight buy-and-hold portfolio -- invest equally in all 23 stocks on day 1, never trade again. This returned 1,326% with a Sharpe of 1.085.

The strategy's 680% return underperforms the benchmark. The strategy has lower drawdown (-23.7% vs -35.1%) but the reduced volatility doesn't compensate for the return gap.

### Key concept: Performance metrics

| Metric | Formula | What it tells you |
|---|---|---|
| **Total Return** | `final_equity / initial_equity - 1` | How much money you made, as a percentage |
| **Annualized Return** | `(1 + total_return) ^ (1/years) - 1` | Average yearly return, compounded |
| **Sharpe Ratio** | `mean(excess_daily_returns) / std(daily_returns) * sqrt(252)` | Return per unit of risk. Higher = better. Uses risk_free_rate = 0 (placeholder) |
| **Max Drawdown** | `min((equity - running_max) / running_max)` | Worst peak-to-trough loss. How much you could have lost at the worst moment |

**Important**: Sharpe and drawdown measure different things. Sharpe measures average volatility. Drawdown measures the worst single episode. A strategy can improve on one and worsen on the other -- they are complementary, not redundant.

---

## 4. Stage 3: Risk Engine

### Purpose

Estimate how the 23 stocks move together (covariance matrix) and decompose portfolio risk into each stock's contribution. This is the foundation for portfolio optimization (coming in Stage 3b).

### Part 1: Covariance estimation (covariance.py)

#### What is a covariance matrix?

A 23x23 matrix where:
- **Diagonal** entries are each stock's **variance** (how much it moves on its own)
- **Off-diagonal** entries are **covariances** between pairs (how much they move together)

If TCS and INFY tend to go up on the same days and down on the same days, their covariance is positive and high. If TCS goes up when CIPLA goes down, their covariance is negative.

#### Why not just use the raw sample covariance?

With 23 stocks and 252 daily observations, you're estimating 276 unique parameters (23x24/2). The raw sample covariance **overfits** -- it finds spurious patterns in the specific 252-day window that won't persist. Extreme correlations get exaggerated.

#### How Ledoit-Wolf shrinkage fixes this

**The formula**:
```
Sigma_shrunk = alpha * Target + (1 - alpha) * Sigma_sample
```

- `Sigma_sample`: the raw 23x23 covariance matrix (noisy but unbiased)
- `Target`: a simpler matrix (scaled identity -- assumes stocks are uncorrelated)
- `alpha`: shrinkage intensity, estimated from the data itself (NOT hand-picked)

**What it does to the numbers**:
- Off-diagonal entries (covariances) are pulled toward zero by the factor `(1-alpha)`. In our case, about -9.76% shrinkage.
- Diagonal entries (variances) are pulled toward the average variance. Low-vol stocks get pulled up, high-vol stocks get pulled down.
- The result is still a valid covariance matrix (symmetric, positive semi-definite) -- just less noisy.

**Why -9.76% specifically**: The shrinkage target has zeros off-diagonal, so every covariance gets multiplied by `(1-alpha)`. With alpha=0.094, that's `1-0.094 = 0.906`, meaning a ~9.4% reduction. The extra 0.4% comes from sklearn using biased covariance (divides by N=252) while pandas uses unbiased (divides by N-1=251). Combined: `0.906 * 251/252 = 0.902`, which is -9.76%.

**The key insight**: alpha is NOT a hyperparameter you tune. The Ledoit-Wolf method estimates it analytically from the data. Higher alpha = noisier data needs more regularization. Our alpha ranged from 0.034 to 0.320 across 16 years (mean 0.096).

#### Step-by-step process

1. Given an "as-of" date, find the trailing 252 trading days from the calendar
2. **Fail loudly** if fewer than 252 days exist (no silent fallback to shorter windows)
3. Pull the 253x23 close-price matrix (253 prices produce 252 returns)
4. Compute simple percentage returns (`price_t / price_{t-1} - 1`), matching Stage 2's convention
5. Apply Ledoit-Wolf shrinkage (via sklearn)
6. Return the shrunk matrix plus metadata (as-of date, window dates, shrinkage intensity)

### Part 2: Rolling wrapper (risk_engine.py)

Instead of computing one covariance matrix, compute one every month across the full 16-year history.

- **Rebalance schedule**: first trading day of each calendar month (deliberate choice, documented)
- **Result**: 200 monthly dates, 187 estimates computed, 13 skipped (Jan 2010 - Jan 2011, insufficient history)
- **Why 13 skipped**: Data starts 2010-01-04. Need 253 price days. The 253rd trading day falls on 2011-01-05. So Jan 2011's rebalance date (2011-01-03, with only 251 days) is the last skip. Feb 2011 (271 days) is the first success.
- **Caching**: estimates stored in a dictionary keyed by date, so any optimizer can look up "give me the covariance matrix that was live on date X" without recomputing

### Part 3: Risk decomposition (MCR / CTR)

Given a covariance matrix and a set of portfolio weights, decompose total portfolio risk into each stock's contribution.

#### The formulas

```
Portfolio variance:      sigma_p^2 = w' * Sigma * w
Portfolio volatility:    sigma_p   = sqrt(sigma_p^2)
Marginal Contribution:   MCR_i     = (Sigma * w)_i / sigma_p
Contribution to Risk:    CTR_i     = w_i * MCR_i
```

**What these mean in plain English**:
- **MCR** (Marginal Contribution to Risk): "If I add 1% more weight to stock i, how much does total portfolio risk change?" High MCR = adding more of this stock increases risk a lot.
- **CTR** (Contribution to Risk): "How much of the total portfolio risk is stock i currently responsible for?" High CTR = this stock is a big risk driver in the current portfolio.

#### The verification identity

`sum(CTR_i for all stocks) = portfolio_volatility`

This is a hard mathematical identity (Euler's theorem for homogeneous functions). If it doesn't hold, there's a bug in the code, not an approximation error. Our implementation achieves this to within 1.04e-17 (machine epsilon).

#### What the results showed (equal-weight portfolio, COVID window)

- Portfolio annualized volatility: 27.1% (elevated due to COVID window 2019-06 to 2020-06)
- Long-run (2015-2025): 15.1% portfolio vol vs 27.1% average individual stock vol -- 1.79x diversification ratio
- Top risk contributors: AXISBANK (7.3%), ICICIBANK (6.4%) -- banks are highly correlated with each other
- Lowest risk: CIPLA (2.6%), DRREDDY (2.8%) -- defensive pharma stocks

#### COVID shrinkage behavior (verified independently)

| Period | Shrinkage intensity |
|---|---|
| 2019 calm period | 0.094 - 0.109 |
| Feb 2020 | 0.118 |
| Mar 2020 | 0.113 |
| Apr 2020 | 0.123 |

The spike is real but modest. Why: the trailing 252-day window means the COVID crash (~20 extreme trading days) is diluted by ~230 calm days from 2019. Shrinkage intensity responds to the whole window's estimation noise, not just the tail. A shorter lookback window would show a sharper response.

### Verification checks performed

| Check | Method | Result |
|---|---|---|
| Matrix symmetry | `max(abs(Sigma - Sigma.T))` | 0.00 (exact) |
| Positive diagonal | All variances > 0 | Verified |
| Positive semi-definite | Min eigenvalue check | 1.49e-04 (all positive) |
| Spot-check vs raw cov | Compare specific pairs before/after shrinkage | Confirmed -9.76% uniform off-diagonal shrinkage |
| CTR identity | `sum(CTR) == sigma_p` | Error: 1.04e-17 (machine epsilon) |
| Fail-loud | Request covariance for 2010-06-01 | Correctly raises ValueError |
| Skip boundary | 2011-01-03 has 251 days (need 253) | Correctly skipped |
| Vol gut check | Compare against long-run portfolio vol | 27.1% is window-specific (COVID); long-run is 15.1% |

---

## 5. Key Results

### Stage 2 Backtest (16 years, 2010-2026)

| Metric | Strategy (MA + RSI) | With Phase 1.5 | Benchmark (Equal-Weight B&H) |
|---|---|---|---|
| Total Return | 680% | 647% | 1,326% |
| Sharpe Ratio | 0.870 | 0.882 | 1.085 |
| Max Drawdown | -23.7% | -27.7% | -35.1% |
| Phase 1.5 swaps | 0 | 304 | N/A |

**Key finding**: The active strategy underperforms passive buy-and-hold on total return but has significantly lower drawdown. The conviction-replacement experiment (304 swaps) showed diversification gain doesn't cover 30bps transaction costs.

### Stage 3 Risk Engine

| Metric | Value |
|---|---|
| Covariance estimates computed | 187 (out of 200 monthly dates) |
| Shrinkage intensity range | 0.034 - 0.320 (mean 0.096) |
| Equal-weight portfolio vol (long-run) | 15.1% annualized |
| Diversification ratio | 1.79x |
| CTR identity error | 1.04e-17 |

---

## 6. File Reference

### Root directory -- Stage 1 (Data Pipeline)

| File | Purpose |
|---|---|
| `data_access.py` | THE data gateway. Only interface to quant.db. Every other file imports from here. |
| `download_nifty_data.py` | Downloads OHLCV from Yahoo Finance for all 23 tickers |
| `load_to_db.py` | Loads downloaded data into SQLite, applies schema |
| `reconcile_calendar.py` | Builds trading_calendar table (intersection of all tickers' trading dates) |
| `fix_stale_rows.py` | Detects and tags forward-filled/stale price rows |
| `gap_check.py` | Scans for missing dates in the data |
| `final_validation.py` | Comprehensive data integrity check |
| `debug_tickers.py` | Ad-hoc debugging during development |
| `test_db.py` | Database schema and query tests |
| `generate_summary.py` | Dataset summary statistics |
| `db/quant.db` | The SQLite database (~12 MB) |
| `db/schema.sql` | Database schema definition (5 tables) |

### Root directory -- Stage 2 (Backtester)

| File | Purpose |
|---|---|
| `config.py` | All tunable parameters in one place. Fill timing, cost model, position sizing, replacement guardrails. |
| `data_feed.py` | Streams historical bars one date at a time. Enforces "no future data" -- current_bar() and history() only expose data up to today. |
| `strategy.py` | Two strategy classes: MovingAverageCrossoverStrategy (20/50 SMA) and RSIMeanReversionStrategy (14-day RSI). Each emits BUY/SELL/HOLD + a conviction score. |
| `ranking.py` | Converts raw strategy scores into 0-1 percentile ranks so MA and RSI scores are comparable. |
| `portfolio.py` | The brain. Owns cash, positions, equity curve. Three-phase signal resolution: exits, conviction replacement, new entries. Inverse-vol sizing. |
| `execution_handler.py` | Turns OrderRequests into Fills. Applies 15bps per-leg cost. |
| `engine.py` | The event loop. Connects DataFeed, both Strategies, Portfolio, and ExecutionHandler. Enforces the 1-bar signal-to-fill lag. |
| `benchmark.py` | Builds an equal-weight buy-and-hold benchmark for comparison. |
| `performance_analyzer.py` | Computes Sharpe, return, drawdown, strategy-vs-benchmark comparison. |
| `run_backtest.py` | Entry point. Configures and runs the full 16-year backtest, prints results, saves equity curves. |

### Root directory -- Diagnostic scripts

| File | Purpose |
|---|---|
| `diagnose_unheld.py` | Checked why AXISBANK/HINDUNILVR/LT were never held. Found: ranking worked fine, sizing was the bottleneck (int division to 0 shares). |
| `diagnose_replacements.py` | Counted Phase 1.5 swaps (304 total) and verified SUNPHARMA's complete trade history. |
| `test_short_backtest.py` | Quick Q1 2020 sanity test to verify the loop works before the full 16-year run. |

### stage3/ directory -- Stage 3 (Risk Engine)

| File | Purpose |
|---|---|
| `__init__.py` | Package init |
| `covariance.py` | Point-in-time Ledoit-Wolf covariance estimator. Given a date, returns the shrunk 23x23 matrix plus metadata. |
| `risk_engine.py` | Rolling monthly wrapper (calls covariance.py 187 times) + MCR/CTR risk decomposition. |
| `verify_risk_engine.py` | Automated verification: symmetry, PSD, spot-check, CTR identity, fail-loud. |
| `verify_independent.py` | Independent hand-verification of 4 specific claims from the automated check. |

### Output files

| File | Purpose |
|---|---|
| `stage2_strategy_equity_curve.csv` | Daily equity values for the strategy (4,109 rows) |
| `stage2_benchmark_equity_curve.csv` | Daily equity values for the buy-and-hold benchmark |

---

## 7. Decision Log

Every deliberate design choice, with the reasoning.

### Stage 1 decisions

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Data source | Yahoo Finance (yfinance) | Free, covers NSE, provides adjusted prices |
| 2 | Price adjustment | auto_adjust=True | Handles splits/bonuses automatically, avoids manual corporate action logic |
| 3 | Calendar construction | Intersection of all tickers' dates | Ensures a rectangular matrix -- every ticker has data on every calendar day |
| 4 | Missing data handling | Tag, don't fix silently | Forward-filled rows are tagged data_quality='forward_filled', never silently interpolated |
| 5 | Data access pattern | Single data_access.py gateway | No other file touches the DB -- single point of change if schema evolves |

### Stage 2 decisions

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 6 | Signal-to-fill lag | 1 bar (signal on Close, fill on next Open) | Prevents look-ahead bias -- you can't trade at a price you only know after market close |
| 7 | Cost model | 30bps round-trip flat | Blended estimate for NSE large-caps (brokerage + STT + stamp + slippage). Conservative but defensible. |
| 8 | Capital pool | Shared (both strategies compete) | More realistic than separate pools. Forces genuine ranking of which trades are best. |
| 9 | Position sizing | Inverse-volatility weighting | Equalizes risk contribution per position. Prevents volatile stocks from dominating the portfolio. |
| 10 | Cross-strategy ranking | Percentile normalization | MA scores and RSI scores are on different scales -- percentiles make them comparable |
| 11 | Same-ticker tie breaking | MA wins ties | Deliberate choice. Trend-following gets priority over mean-reversion on exact ties. |
| 12 | Exit: signal-driven | SELL = full exit | When the owning strategy explicitly says sell, don't second-guess it |
| 13 | Exit: scale-out | First BUY-to-HOLD = sell 50% | Locks in some profit while keeping exposure. One-time trigger only. |
| 14 | Exit: replacement | Full exit + full entry | Conviction replacement is a clean swap, not a partial trim. Different semantics from scale-out. |
| 15 | Replacement edge | 0.40 percentile gap | Must be meaningfully stronger to justify the 30bps round-trip cost |
| 16 | Holding period guard | 10 bars minimum | Prevents thrashing -- don't displace yesterday's buy |
| 17 | Replacement cap | 2 per day | Limits turnover even when the edge threshold is met |
| 18 | New entries per day | top_k = 5 | Limits how many new positions open in a single day |
| 19 | Max deployment | 95% of equity | 5% cash buffer for slippage/costs. Not a return drag -- it's insurance. |
| 20 | Benchmark | Equal-weight buy-and-hold | Simple, transparent, uses the same universe. More honest than comparing to a random index. |
| 21 | Risk-free rate | 0% (placeholder) | Overstates Sharpe. Real Indian T-bill rate (~6-7%) would lower both strategy and benchmark Sharpe. |
| 22 | Whole shares only | int(allocation // price) | NSE doesn't allow fractional shares in cash equity. This causes small-amount orders to round to 0. |

### Stage 3 decisions

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 23 | Return convention | Simple returns (p_t/p_{t-1} - 1) | Matches Stage 2's pct_change() in performance_analyzer.py. One convention across the project. |
| 24 | Forward-filled rows | Include as-is | Only 28 out of 94,500 rows (0.03%). Immaterial to a 252-day window. Deliberate, documented. |
| 25 | Trailing window | 252 trading days | ~1 year. Standard institutional convention. |
| 26 | Insufficient history | Fail loudly (ValueError) | Mirrors Stage 2's discipline. No silent fallback to shorter windows. |
| 27 | Shrinkage method | Ledoit-Wolf (sklearn) | Data-driven alpha (no hand-tuning). Standard for moderate-dimensional problems (p/n < 1). |
| 28 | Rebalance frequency | Monthly (first trading day) | Daily too costly, annual misses regimes. First trading day is a deliberate, documented choice. |
| 29 | Covariance caching | Dictionary keyed by date | Avoids recomputing -- same matrix is reused by both optimizers later. |

---

## 8. Concept Glossary

Quick reference for every concept used in the project.

| Concept | Plain English |
|---|---|
| **SMA (Simple Moving Average)** | Average of the last N closing prices. SMA(20) = average of last 20 closes. Smooths out daily noise to show the trend. |
| **RSI (Relative Strength Index)** | 0-100 scale measuring recent price momentum. Below 30 = "oversold" (stock may bounce). Above 70 = "overbought" (stock may drop). |
| **Look-ahead bias** | Using information from the future to make decisions in the past. A fatal flaw in backtesting. Example: using today's closing price to decide to buy at that same closing price. |
| **Backtesting** | Simulating a trading strategy on historical data to see how it would have performed. Not a prediction of future results. |
| **Equity curve** | A time series of your portfolio's total value. Starts at initial_cash, goes up or down based on trades and market movements. |
| **Drawdown** | How much your portfolio drops from its peak. A 20% drawdown means you are 20% below your highest-ever value. |
| **Sharpe ratio** | Return per unit of risk. (average excess return) / (volatility of returns). Higher = better risk-adjusted performance. Annualized by multiplying by sqrt(252). |
| **Basis point (bps)** | 1/100th of a percent. 30bps = 0.30%. 15bps per leg = 0.15% cost on each buy or sell trade. |
| **Inverse-volatility weighting** | Give more money to low-volatility stocks, less to high-volatility ones. Makes each position contribute roughly equal risk. |
| **Covariance matrix** | A square matrix capturing how all stocks move relative to each other. Diagonal = individual variances. Off-diagonal = pairwise covariances. |
| **Positive semi-definite (PSD)** | A mathematical property every valid covariance matrix must have. Guarantees portfolio variance is never negative. If a matrix isn't PSD, it can produce nonsensical "negative risk" results. |
| **Shrinkage** | Blending a noisy estimate with a simpler, more stable estimate to reduce overfitting. Like averaging between what you observed and what you expected. |
| **Ledoit-Wolf** | A specific shrinkage method that automatically determines how much to blend (the alpha). You don't pick alpha -- it is estimated from the data. |
| **MCR (Marginal Contribution to Risk)** | How much portfolio risk changes if you add a small amount more of stock i. Units: same as portfolio volatility. |
| **CTR (Contribution to Risk)** | How much total portfolio risk stock i is currently responsible for. CTR_i = weight_i * MCR_i. All CTRs sum to portfolio vol. |
| **Risk parity** | An allocation method where every stock contributes the same amount of risk (equal CTR). Not implemented yet -- coming in Stage 3b. |
| **Percentile rank** | Converts raw scores to a 0-1 scale based on relative position. If you rank 4th out of 20, your percentile is 4/20 = 0.20. |
| **Forward-fill** | When a price is missing, use the previous day's closing price. Creates a zero-return day for that stock. |
| **Rectangular matrix** | A data matrix with no missing values -- every row (date) has data for every column (ticker). Required for clean iteration. |
| **Diversification ratio** | Average individual stock vol / portfolio vol. Higher = more diversification benefit. Our 1.79x means the portfolio is 44% less volatile than the average stock. |
| **Signal-to-fill lag** | The delay between deciding to trade and actually executing. In our system: 1 bar. Prevents look-ahead bias. |
| **Mark to market** | Recording the current value of all positions using today's prices. Done at the end of every bar to build the equity curve. |
