"""
Binance Vision downloader — pure module, no CLI.
Call download_from_vision(symbol, interval, days_back, timezone) to get a DataFrame.
Supports spot and perpetual futures. Parallel monthly download + API fallback for current month.
"""

import requests
import zipfile
import io
import time
import threading
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytz

# ── Download constants ──
TIMEOUT     = 300   # Seconds before timeout
RETRIES     = 3     # Retry attempts on timeout
RETRY_DELAY = 5     # Seconds between retries
API_LIMIT   = 1500  # Max candles per Binance API call

_print_lock = threading.Lock()


def _tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def _format_size(bytes_total):
    if bytes_total >= 1_000_000:
        return f"{bytes_total / 1_000_000:.1f} MB"
    elif bytes_total >= 1_000:
        return f"{bytes_total / 1_000:.1f} KB"
    return f"{bytes_total} B"


def _download_month_stream(url, label=""):
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, stream=True, timeout=(10, 30))

            if r.status_code != 200:
                return r

            total      = int(r.headers.get('content-length', 0))
            downloaded = 0
            chunks     = []
            start_time = time.time()

            for chunk in r.iter_content(chunk_size=1048576):
                if chunk:
                    chunks.append(chunk)
                    downloaded += len(chunk)
                    elapsed = time.time() - start_time
                    speed   = downloaded / elapsed if elapsed > 0 else 0
                    if total:
                        pct = downloaded / total * 100
                        _tprint(f"  {label}  {_format_size(downloaded)} / {_format_size(total)} ({pct:.0f}%)  {_format_size(speed)}/s")
                    else:
                        _tprint(f"  {label}  {_format_size(downloaded)} downloaded  {_format_size(speed)}/s")

            r._content = b"".join(chunks)
            return r

        except requests.exceptions.Timeout:
            _tprint(f"  {label} Timeout (attempt {attempt+1}/{RETRIES}), retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
        except requests.exceptions.ChunkedEncodingError:
            _tprint(f"  {label} Connection dropped (attempt {attempt+1}/{RETRIES}), retrying...")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            _tprint(f"  {label} Error: {e}")
            return None

    _tprint(f"  {label} Failed after all retries")
    return None


def _parse_vision_csv(z):
    raw        = z.open(z.namelist()[0])
    first_line = raw.readline().decode('utf-8').strip()
    raw.seek(0)
    has_header = not first_line.split(',')[0].strip().lstrip('-').isdigit()
    df = pd.read_csv(raw, header=0 if has_header else None)
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                  'close_time', 'quote_volume', 'trades',
                  'taker_buy_base', 'taker_buy_quote', 'ignore']
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]


def _download_via_api(clean_symbol, interval, api_url, start_ms, end_ms):
    all_candles = []
    current     = start_ms
    batch       = 0

    while current < end_ms:
        try:
            params = {
                "symbol":    clean_symbol,
                "interval":  interval,
                "startTime": current,
                "limit":     API_LIMIT,
            }
            r    = requests.get(api_url, params=params, timeout=30)
            data = r.json()

            if not data or isinstance(data, dict):
                break

            filtered = [c for c in data if c[0] <= end_ms]
            all_candles.extend(filtered)
            batch += 1
            print(f"\r  {len(all_candles):,} candles fetched (batch {batch})...", end="", flush=True)

            current = data[-1][0] + 1
            if len(data) < API_LIMIT:
                break

        except Exception as e:
            print(f"\n  API error: {e}")
            break

    print()

    if not all_candles:
        return None

    df = pd.DataFrame(all_candles, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]


def download_from_vision(
    symbol:    str = "ETHUSDT.P",
    interval:  str = "3m",
    days_back: int = 365,
    timezone:  str = "UTC",
) -> pd.DataFrame | None:
    """
    Download OHLCV data from Binance Vision (completed months) + Binance API (current month).

    Returns a DataFrame with columns [timestamp, open, high, low, close, volume]
    where timestamp is a naive datetime in the given timezone, or None on failure.
    """
    is_perp      = symbol.endswith(".P")
    clean_symbol = symbol.replace(".P", "")
    vision_url   = (
        "https://data.binance.vision/data/futures/um/monthly/klines"
        if is_perp else
        "https://data.binance.vision/data/spot/monthly/klines"
    )
    api_url = (
        "https://fapi.binance.com/fapi/v1/klines"
        if is_perp else
        "https://api.binance.com/api/v3/klines"
    )

    tz       = pytz.timezone(timezone)
    now      = datetime.now(tz)
    start_dt = now - timedelta(days=days_back)

    print("=" * 60)
    print(f"  Symbol   : {symbol}  ({interval})")
    print(f"  From     : {start_dt.strftime('%Y-%m-%d %H:%M')} {timezone}")
    print(f"  To       : {now.strftime('%Y-%m-%d %H:%M')} {timezone}")
    print(f"  Days     : {days_back}")
    print("=" * 60)

    # Build list of (year, month) needed
    months_needed = []
    cur = start_dt.replace(day=1)
    while cur <= now:
        months_needed.append((cur.year, cur.month))
        cur = cur.replace(month=cur.month + 1) if cur.month < 12 else cur.replace(year=cur.year + 1, month=1)

    completed_months = [(y, m) for y, m in months_needed if not (y == now.year and m == now.month)]
    current_month    = [(y, m) for y, m in months_needed if y == now.year and m == now.month]

    print(f"\nMonths needed: {months_needed}")
    workers = max(1, len(completed_months))
    print(f"Downloading {len(completed_months)} months with {workers} parallel workers...\n")

    results = {}

    def fetch_month(year, month):
        label = f"{year}-{str(month).zfill(2)}"
        url   = (
            f"{vision_url}/{clean_symbol}/{interval}/"
            f"{clean_symbol}-{interval}-{year}-{str(month).zfill(2)}.zip"
        )
        _tprint(f"  {label}  Downloading...")

        r = _download_month_stream(url, label=label)

        if r is None:
            _tprint(f"  {label}  Failed\n")
            return year, month, None

        if r.status_code == 404:
            _tprint(f"  {label}  Not on Vision — trying API...\n")
            month_start = datetime(year, month, 1, tzinfo=pytz.UTC)
            month_end   = (
                datetime(year + 1, 1, 1, tzinfo=pytz.UTC) if month == 12
                else datetime(year, month + 1, 1, tzinfo=pytz.UTC)
            )
            df = _download_via_api(clean_symbol, interval, api_url,
                                   int(month_start.timestamp() * 1000),
                                   int(month_end.timestamp()   * 1000))
            if df is not None:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df['timestamp'] = df['timestamp'].dt.tz_convert(timezone).dt.tz_localize(None)
                _tprint(f"  {label}  {len(df):,} candles (via API)\n")
                return year, month, df
            _tprint(f"  {label}  API fallback also failed\n")
            return year, month, None

        if r.status_code != 200:
            _tprint(f"  {label}  HTTP {r.status_code}\n")
            return year, month, None

        try:
            z  = zipfile.ZipFile(io.BytesIO(r.content))
            df = _parse_vision_csv(z)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['timestamp'] = df['timestamp'].dt.tz_convert(timezone).dt.tz_localize(None)
            _tprint(f"  {label}  {len(df):,} candles\n")
            return year, month, df
        except Exception as e:
            _tprint(f"  {label}  Parse error: {e}\n")
            return year, month, None

    # ── Parallel download of completed months ──
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_month, y, m): (y, m) for y, m in completed_months}
        for future in as_completed(futures):
            year, month, df = future.result()
            if df is not None:
                results[(year, month)] = df

    # ── API fallback for current month ──
    for year, month in current_month:
        _tprint(f"  {year}-{str(month).zfill(2)} (current month — using API)")
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        start_ms    = int(month_start.astimezone(pytz.UTC).timestamp() * 1000)
        end_ms      = int(now.astimezone(pytz.UTC).timestamp() * 1000)
        df = _download_via_api(clean_symbol, interval, api_url, start_ms, end_ms)
        if df is not None:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['timestamp'] = df['timestamp'].dt.tz_convert(timezone).dt.tz_localize(None)
            results[(year, month)] = df
            _tprint(f"  {len(df):,} candles\n")

    frames = [results[k] for k in sorted(results.keys()) if k in results]
    if not frames:
        print("No data downloaded!")
        return None

    print("Merging all months...")
    final      = pd.concat(frames).drop_duplicates(subset='timestamp').sort_values('timestamp')
    start_naive = start_dt.replace(tzinfo=None)
    now_naive   = now.replace(tzinfo=None)
    final       = final[(final['timestamp'] >= start_naive) & (final['timestamp'] <= now_naive)]
    final       = final.reset_index(drop=True)

    print(f"Total candles: {len(final):,}  |  {final['timestamp'].min()} → {final['timestamp'].max()}")
    return final
