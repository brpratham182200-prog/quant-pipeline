download_nifty_data.py
1. Ticker definitions (TICKER_CATEGORIES)

Instead of one flat list, tickers are grouped by sector in a dictionary:

python
TICKER_CATEGORIES = {"Banking & Financial Services": [...], "IT": [...], ...}

This is a nice upgrade over a flat list — it lets you generate sector-level summaries later (useful once you get to portfolio diversification in stage 3). ALL_TICKERS flattens this into the list yf.download actually needs.

2. Download configuration (DOWNLOAD_CONFIG)
python
"auto_adjust": True,   # split/dividend adjusted
"group_by": "ticker",
"threads": True,

Same config we discussed — auto_adjust=True handles the split/dividend problem, group_by="ticker" gives you a MultiIndex DataFrame keyed by ticker first (so data["TCS.NS"] gets you that stock's OHLCV), and threads=True parallelizes the download requests so it doesn't fetch 22 tickers sequentially.

3. Output directory setup
python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR / PER_TICKER_DIR / SUMMARY_DIR

Creates a data/ folder next to the script with three subfolders: raw/ (the full pickled dataset), per_ticker/ (individual CSVs), summary/ (stats). os.makedirs(..., exist_ok=True) means re-running the script won't error if folders already exist.

4. download_data()

Calls yf.download() with your config, prints a readable header (ticker count, date range, timestamp), and returns the raw MultiIndex DataFrame. Nothing fancy here — it's a thin wrapper for logging/readability, which matters more than it sounds once you're debugging why a run failed at 2am.

5. save_raw(data)

Pickles the entire multi-index DataFrame to disk. Pickle preserves the exact DataFrame structure (dtypes, MultiIndex) — unlike CSV, which would need re-parsing. This is your "source of truth" backup — if per-ticker export logic has a bug later, you can always reload from this pickle without re-hitting Yahoo Finance.

6. save_per_ticker_csv(data)

This is the meatiest function. For each ticker:

python
ticker_df = data[ticker].copy()
ticker_df = ticker_df.dropna(how="all")

Extracts that ticker's slice from the MultiIndex, drops rows where every column is NaN (this handles tickers with slightly different listing dates — e.g. if one stock IPO'd in 2012, its 2010-2011 rows would be all-NaN and get dropped here).

python
safe_name = ticker.replace(".", "_").replace("&", "and")

Sanitizes the filename — important because M&M.NS would otherwise produce a filesystem-unfriendly filename.

Then it prints a preview (head/tail, date range, row count) per ticker, wrapped in a try/except KeyError — this catches the case where yfinance silently fails to return data for a ticker (delisted, wrong symbol, temporary API issue) without crashing your whole run.

7. generate_summary(data)

Builds one row per ticker with: trading day count, date range, min/max/mean/latest close, and total volume. This becomes your data_summary.csv — a quick sanity-check table you (or an interviewer) can scan to see "does this data look reasonable?" without opening 22 individual files.

8. Main execution block

Runs everything in order: download → save raw pickle → export per-ticker CSVs → generate summary → print a final confirmation with file paths.

In finance and trading, a ticker (or ticker symbol) is a unique combination of letters assigned to a publicly traded security (like stocks or ETFs) for quick identification on an exchange.

Purpose: Acts as a short, unambiguous code so traders and automated systems don't confuse companies with similar names.

Exchange Formats:

US Markets (NYSE / NASDAQ): Typically 1 to 4 letters, like AAPL (Apple), MSFT (Microsoft), or F (Ford).

Indian Markets (NSE): Typically the full company name or abbreviation followed by the exchange code, such as TCS.NS (Tata Consultancy Services) or HDFCBANK.NS (HDFC Bank).

Historical Origin: The term comes from the historical ticker tape machines that printed stock symbols and price updates onto long strips of paper, making distinct short codes necessary for fast transmission.