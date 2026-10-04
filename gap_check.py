"""
================================================================================
Trading Calendar Gap Check
================================================================================
Compares each ticker's actual trading dates against the official NSE trading
calendar to distinguish expected non-trading days (weekends, market holidays)
from genuine data gaps (missing weekday data = potential pipeline/feed issue).
================================================================================
"""
import pandas as pd

try:
    import pandas_market_calendars as mcal
    HAS_MCAL = True
except ImportError:
    HAS_MCAL = False
    print("[WARNING] pandas_market_calendars not installed.")
    print("  Install with: pip install pandas_market_calendars")
    print("  Falling back to business-day approximation (won't know real holidays).\n")

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
# 2. BUILD THE EXPECTED TRADING CALENDAR
# =============================================================================
def get_expected_trading_days(start, end):
    """
    Returns the set of dates the NSE was actually open for trading,
    using the official exchange calendar if available.
    """
    if HAS_MCAL:
        nse = mcal.get_calendar("NSE")
        schedule = nse.schedule(start_date=start, end_date=end)
        return set(schedule.index.date)
    else:
        # Fallback: assumes only weekends are closed (misses actual holidays,
        # so it will under-report gaps -- treat results as approximate)
        bdays = pd.bdate_range(start=start, end=end)
        return set(bdays.date)

# =============================================================================
# 3. GAP CHECK PER TICKER
# =============================================================================
print("=" * 70)
print("TRADING CALENDAR GAP CHECK")
if HAS_MCAL:
    print("  Using: NSE official exchange calendar (pandas_market_calendars)")
else:
    print("  Using: Business-day approximation (holidays not excluded)")
print("=" * 70)

gap_summary = []

for sector, tickers in TICKER_CATEGORIES.items():
    for ticker in tickers:
        try:
            df = raw[ticker].dropna(how="all")
            if len(df) == 0:
                gap_summary.append({
                    "Ticker": ticker,
                    "Sector": sector,
                    "Expected Trading Days": 0,
                    "Actual Rows": 0,
                    "Missing Days": 0,
                    "Status": "NO DATA",
                })
                print(f"\n{ticker}: SKIPPED (no data)")
                continue

            actual_dates = set(df.index.date)
            start, end = df.index.min(), df.index.max()
            expected_dates = get_expected_trading_days(start, end)

            # Dates the exchange was open, but this ticker has no row for
            missing_dates = sorted(expected_dates - actual_dates)

            # Also check: dates the ticker HAS data for but exchange was "closed"
            # (extra dates could indicate data from special trading sessions)
            extra_dates = sorted(actual_dates - expected_dates)

            gap_summary.append({
                "Ticker": ticker,
                "Sector": sector,
                "Expected Trading Days": len(expected_dates),
                "Actual Rows": len(actual_dates),
                "Missing Days": len(missing_dates),
                "Extra Days": len(extra_dates),
                "Status": "CLEAN" if len(missing_dates) == 0 else "GAPS FOUND",
            })

            if missing_dates:
                print(f"\n{ticker}: {len(missing_dates)} missing trading day(s)")
                # Show first 10 to avoid flooding console on a bad ticker
                for d in missing_dates[:10]:
                    print(f"  MISSING: {d}")
                if len(missing_dates) > 10:
                    print(f"  ... and {len(missing_dates) - 10} more")

            if extra_dates:
                print(f"\n{ticker}: {len(extra_dates)} extra day(s) (data on exchange holidays?)")
                for d in extra_dates[:5]:
                    print(f"  EXTRA: {d}")
                if len(extra_dates) > 5:
                    print(f"  ... and {len(extra_dates) - 5} more")

        except Exception as e:
            print(f"{ticker}: Error during gap check - {e}")

# =============================================================================
# 4. SUMMARY TABLE
# =============================================================================
summary_df = pd.DataFrame(gap_summary)
summary_df.to_csv("data/summary/gap_check_summary.csv", index=False)

print("\n" + "=" * 70)
print("GAP CHECK SUMMARY")
print("=" * 70)
print(summary_df.to_string(index=False))

valid = summary_df[summary_df["Status"] != "NO DATA"]
clean_tickers = len(valid[valid["Missing Days"] == 0])
total_valid = len(valid)
print(f"\n{clean_tickers} / {total_valid} tickers have zero unexplained gaps.")

if len(valid[valid["Missing Days"] > 0]) > 0:
    print("\nTickers with gaps:")
    for _, row in valid[valid["Missing Days"] > 0].iterrows():
        print(f"  {row['Ticker']:20s}  {row['Missing Days']:4d} missing days")

print("\nSaved to data/summary/gap_check_summary.csv")
print("=" * 70)
