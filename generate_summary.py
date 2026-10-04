"""
================================================================================
Data Summary + Sanity Check Script
================================================================================
Generates summary statistics from the raw pickle data and runs a sanity check
for suspicious single-day price moves (>20%), which could indicate data quality
issues such as unadjusted splits or bad data points.
================================================================================
"""
import pandas as pd

# =============================================================================
# 1. LOAD RAW DATA
# =============================================================================
raw = pd.read_pickle("data/raw/nifty_subset_raw.pkl")

TICKER_CATEGORIES = {
    "Banking & Financial Services": ["HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","KOTAKBANK.NS","AXISBANK.NS"],
    "Information Technology": ["TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS"],
    "Energy & Power": ["RELIANCE.NS","ONGC.NS","NTPC.NS"],
    "FMCG / Consumer Staples": ["HINDUNILVR.NS","ITC.NS","NESTLEIND.NS"],
    "Pharmaceuticals": ["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS"],
    "Automobiles": ["MARUTI.NS","TATAMOTORS.NS","M&M.NS"],
    "Infrastructure & Cement": ["LT.NS","ULTRACEMCO.NS"],
}

ALL_TICKERS = [t for tickers in TICKER_CATEGORIES.values() for t in tickers]

# =============================================================================
# 2. GENERATE SUMMARY
# =============================================================================
print("=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

rows = []
for sector, tickers in TICKER_CATEGORIES.items():
    for ticker in tickers:
        try:
            df = raw[ticker].dropna(how="all")
            if len(df) == 0:
                rows.append({"Sector": sector, "Ticker": ticker, "Trading Days": 0,
                             "Start Date": None, "End Date": None, "Status": "UNAVAILABLE"})
                continue
            close = df["Close"].dropna()
            volume = df["Volume"].dropna() if "Volume" in df.columns else pd.Series(dtype=float)
            rows.append({
                "Sector": sector, "Ticker": ticker, "Trading Days": len(df),
                "Start Date": str(df.index.min().date()), "End Date": str(df.index.max().date()),
                "Close Min (INR)": round(float(close.min()), 2),
                "Close Max (INR)": round(float(close.max()), 2),
                "Close Mean (INR)": round(float(close.mean()), 2),
                "Close Latest (INR)": round(float(close.iloc[-1]), 2),
                "Total Volume": int(volume.sum()),
                "Status": "OK"
            })
        except Exception as e:
            rows.append({"Sector": sector, "Ticker": ticker, "Trading Days": 0, "Status": str(e)})

summary_df = pd.DataFrame(rows)
summary_df.to_csv("data/summary/data_summary.csv", index=False)

ok_count = len(summary_df[summary_df["Status"] == "OK"])
print("Summary saved to data/summary/data_summary.csv")
print("Successfully downloaded: %d / %d tickers" % (ok_count, len(summary_df)))
print()

for _, row in summary_df.iterrows():
    status = row["Status"]
    mark = "[OK]" if status == "OK" else "[!!]"
    days = row.get("Trading Days", 0)
    ticker = row["Ticker"]
    sector = row["Sector"]
    if status == "OK":
        latest = row.get("Close Latest (INR)", "N/A")
        print("  %s %-20s %-35s %4d days  Latest: INR %s" % (mark, ticker, sector, days, latest))
    else:
        print("  %s %-20s %-35s %4d days  (%s)" % (mark, ticker, sector, days, status))

# =============================================================================
# 3. SANITY CHECK: Suspicious single-day price moves (>20%)
# =============================================================================
print()
print("=" * 70)
print("SANITY CHECK: Single-day moves > 20%%")
print("=" * 70)
print("(These could indicate unadjusted splits, bad data, or genuine crashes)")
print()

issues_found = False
for ticker in ALL_TICKERS:
    try:
        df = raw[ticker].dropna(how="all")
        if len(df) == 0:
            continue
        close = df["Close"].dropna()
        daily_ret = close.pct_change()
        suspicious = daily_ret[daily_ret.abs() > 0.20]  # >20% single-day move
        if not suspicious.empty:
            issues_found = True
            print("%s: %d suspicious move(s) detected" % (ticker, len(suspicious)))
            for date, ret in suspicious.items():
                date_str = str(date.date()) if hasattr(date, 'date') else str(date)
                direction = "UP" if ret > 0 else "DOWN"
                prev_idx = close.index.get_loc(date) - 1
                if prev_idx >= 0:
                    prev_close = close.iloc[prev_idx]
                    curr_close = close.loc[date]
                    print("  %s: %.1f%% %s (%.2f -> %.2f)" % (
                        date_str, ret * 100, direction, prev_close, curr_close))
                else:
                    print("  %s: %.1f%% %s" % (date_str, ret * 100, direction))
            print()
    except Exception as e:
        print("%s: Error during sanity check - %s" % (ticker, e))

if not issues_found:
    print("  No suspicious moves detected! Data looks clean.")
    print("  (auto_adjust=True is properly handling splits and dividends)")

print()
print("=" * 70)
print("SANITY CHECK COMPLETE")
print("=" * 70)
