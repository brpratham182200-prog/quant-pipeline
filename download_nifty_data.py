"""
================================================================================
NSE (Nifty 50 Subset) Historical Data Downloader
================================================================================

Purpose:
    Downloads daily OHLCV data for a curated subset of Nifty 50 stocks from
    Yahoo Finance, using split/dividend-adjusted prices.

Data Source:
    Yahoo Finance via the `yfinance` library.

Key Configuration:
    - auto_adjust=True: Adjusts Open/High/Low/Close for stock splits and
      dividends, eliminating artificial price jumps ("fake crashes").
    - Date range: 2010-01-01 to 2026-08-31
    - Interval: 1 day

Ticker Categories:
    1. Banking & Financial Services (5 tickers)
    2. Information Technology (4 tickers)
    3. Energy & Power (3 tickers)
    4. FMCG / Consumer Staples (3 tickers)
    5. Pharmaceuticals (3 tickers)
    6. Automobiles (3 tickers)
    7. Infrastructure & Cement (2 tickers)

Output:
    - Raw multi-index DataFrame saved as pickle: data/raw/nifty_subset_raw.pkl
    - Per-ticker CSV files in: data/per_ticker/
    - Summary statistics saved as CSV: data/summary/data_summary.csv

================================================================================
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# =============================================================================
# 1. TICKER DEFINITIONS — Categorised by Sector
# =============================================================================

# Each category maps a sector name to its constituent NSE tickers.
# The .NS suffix is required for Yahoo Finance to resolve NSE-listed stocks.

TICKER_CATEGORIES = {
    "Banking & Financial Services": [
        "HDFCBANK.NS",   # HDFC Bank
        "ICICIBANK.NS",  # ICICI Bank
        "SBIN.NS",       # State Bank of India
        "KOTAKBANK.NS",  # Kotak Mahindra Bank
        "AXISBANK.NS",   # Axis Bank
    ],
    "Information Technology": [
        "TCS.NS",        # Tata Consultancy Services
        "INFY.NS",       # Infosys
        "WIPRO.NS",      # Wipro
        "HCLTECH.NS",    # HCL Technologies
    ],
    "Energy & Power": [
        "RELIANCE.NS",   # Reliance Industries
        "ONGC.NS",       # Oil & Natural Gas Corporation
        "NTPC.NS",       # NTPC Limited
    ],
    "FMCG / Consumer Staples": [
        "HINDUNILVR.NS", # Hindustan Unilever
        "ITC.NS",        # ITC Limited
        "NESTLEIND.NS",  # Nestlé India
    ],
    "Pharmaceuticals": [
        "SUNPHARMA.NS",  # Sun Pharmaceutical
        "DRREDDY.NS",    # Dr. Reddy's Laboratories
        "CIPLA.NS",      # Cipla
    ],
    "Automobiles": [
        "MARUTI.NS",     # Maruti Suzuki
        "TATAMOTORS.NS", # Tata Motors
        "M&M.NS",        # Mahindra & Mahindra
    ],
    "Infrastructure & Cement": [
        "LT.NS",         # Larsen & Toubro
        "ULTRACEMCO.NS", # UltraTech Cement
    ],
}

# Flatten all tickers into a single list for download
ALL_TICKERS = [ticker for tickers in TICKER_CATEGORIES.values() for ticker in tickers]

# =============================================================================
# 2. DOWNLOAD CONFIGURATION
# =============================================================================

DOWNLOAD_CONFIG = {
    "start": "2010-01-01",
    "end": "2026-08-31",
    "interval": "1d",
    "auto_adjust": True,   # Adjusts OHLC for splits & dividends
    "group_by": "ticker",
    "threads": True,
}

# =============================================================================
# 3. OUTPUT DIRECTORY SETUP
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PER_TICKER_DIR = os.path.join(DATA_DIR, "per_ticker")
SUMMARY_DIR = os.path.join(DATA_DIR, "summary")

for directory in [RAW_DIR, PER_TICKER_DIR, SUMMARY_DIR]:
    os.makedirs(directory, exist_ok=True)

# =============================================================================
# 4. DOWNLOAD DATA
# =============================================================================

def download_data():
    """
    Download historical OHLCV data for all tickers from Yahoo Finance.

    Returns:
        pd.DataFrame: MultiIndex DataFrame with (ticker, field) columns.
    """
    print("=" * 70)
    print("DOWNLOADING NSE (Nifty 50 Subset) DATA")
    print("=" * 70)
    print(f"  Tickers    : {len(ALL_TICKERS)} stocks across {len(TICKER_CATEGORIES)} sectors")
    print(f"  Date Range : {DOWNLOAD_CONFIG['start']} -> {DOWNLOAD_CONFIG['end']}")
    print(f"  Interval   : {DOWNLOAD_CONFIG['interval']}")
    print(f"  Auto-Adjust: {DOWNLOAD_CONFIG['auto_adjust']} (split/dividend adjusted)")
    print(f"  Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)

    data = yf.download(
        tickers=ALL_TICKERS,
        **DOWNLOAD_CONFIG,
    )

    print("-" * 70)
    print(f"  Download complete. Shape: {data.shape}")
    print(f"  Date index range: {data.index.min()} -> {data.index.max()}")
    print("=" * 70)
    return data


# =============================================================================
# 5. SAVE & SUMMARISE
# =============================================================================

def save_raw(data):
    """Save the full multi-index DataFrame as a pickle file."""
    path = os.path.join(RAW_DIR, "nifty_subset_raw.pkl")
    data.to_pickle(path)
    print(f"\n[SAVED] Raw data -> {path}")


def save_per_ticker_csv(data):
    """
    Extract each ticker's OHLCV data and save as individual CSV files.
    Also prints head/tail for verification.
    """
    print("\n" + "=" * 70)
    print("PER-TICKER DATA EXPORT")
    print("=" * 70)

    for sector, tickers in TICKER_CATEGORIES.items():
        print(f"\n{'-' * 50}")
        print(f"  Sector: {sector}")
        print(f"{'-' * 50}")

        for ticker in tickers:
            try:
                ticker_df = data[ticker].copy()
                ticker_df = ticker_df.dropna(how="all")  # drop rows with no data

                # Save CSV
                safe_name = ticker.replace(".", "_").replace("&", "and")
                csv_path = os.path.join(PER_TICKER_DIR, f"{safe_name}.csv")
                ticker_df.to_csv(csv_path)

                # Print preview
                print(f"\n  > {ticker} - {len(ticker_df)} trading days")
                print(f"    Date range: {ticker_df.index.min().date()} -> {ticker_df.index.max().date()}")
                print(f"    Saved -> {csv_path}")
                print(f"    Head:")
                print(ticker_df.head(3).to_string().replace("\n", "\n    "))
                print(f"    Tail:")
                print(ticker_df.tail(3).to_string().replace("\n", "\n    "))

            except KeyError:
                print(f"\n  [X] {ticker} - NOT FOUND in downloaded data (skipped)")


def generate_summary(data):
    """
    Generate a summary table with per-ticker statistics:
    - Trading days count
    - Date range
    - Close price: min, max, mean, latest
    - Total volume
    """
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    rows = []
    for sector, tickers in TICKER_CATEGORIES.items():
        for ticker in tickers:
            try:
                df = data[ticker].dropna(how="all")
                close = df["Close"].dropna()
                volume = df["Volume"].dropna() if "Volume" in df.columns else pd.Series(dtype=float)

                rows.append({
                    "Sector": sector,
                    "Ticker": ticker,
                    "Trading Days": len(df),
                    "Start Date": df.index.min().date(),
                    "End Date": df.index.max().date(),
                    "Close Min (₹)": round(close.min(), 2),
                    "Close Max (₹)": round(close.max(), 2),
                    "Close Mean (₹)": round(close.mean(), 2),
                    "Close Latest (₹)": round(close.iloc[-1], 2) if len(close) > 0 else None,
                    "Total Volume": int(volume.sum()) if len(volume) > 0 else None,
                })
            except KeyError:
                rows.append({
                    "Sector": sector,
                    "Ticker": ticker,
                    "Trading Days": 0,
                    "Start Date": None,
                    "End Date": None,
                })

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(SUMMARY_DIR, "data_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(summary_df.to_string(index=False))
    print(f"\n[SAVED] Summary → {summary_path}")
    return summary_df


# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Download
    data = download_data()

    # Save raw pickle
    save_raw(data)

    # Save per-ticker CSVs + preview
    save_per_ticker_csv(data)

    # Generate & save summary
    summary = generate_summary(data)

    print("\n" + "=" * 70)
    print("ALL DONE [OK]")
    print(f"  Raw pickle  : {os.path.join(RAW_DIR, 'nifty_subset_raw.pkl')}")
    print(f"  Per-ticker  : {PER_TICKER_DIR}/ ({len(ALL_TICKERS)} CSVs)")
    print(f"  Summary     : {os.path.join(SUMMARY_DIR, 'data_summary.csv')}")
    print("=" * 70)
