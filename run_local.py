"""
ICT Unicorn — Local trigger for RunPod serverless optimization.

Usage:
  1. Fill in RUNPOD_API_KEY and ENDPOINT_ID below.
  2. Edit the config dict to set your asset, interval, and parameter ranges.
  3. Run:  python run_local.py
  4. Results are saved to a CSV file in this directory.

Install locally first (one-time):
  pip install runpod pandas
"""

import runpod
import base64
import io
import pandas as pd
from datetime import datetime

# ============================================================
#  RUNPOD CREDENTIALS
#  Get these from: https://www.runpod.io/console/serverless
# ============================================================
RUNPOD_API_KEY = "your-api-key-here"       # RunPod → Settings → API Keys
ENDPOINT_ID    = "your-endpoint-id-here"   # RunPod → Serverless → your endpoint

# ============================================================
#  OPTIMIZATION CONFIGURATION
#  Everything you need is in this one config dict.
# ============================================================
config = {

    # ── Data ──────────────────────────────────────────────────────────────
    # Symbol: use ETHUSDT.P for ETH perp, BTCUSDT.P for BTC perp,
    #         ETHUSDT (no .P) for spot.
    "symbol":    "ETHUSDT.P",
    "interval":  "3m",          # Options: 1m, 3m, 5m, 15m, 1h, 4h, 1d
    "days_back": 365,            # How many days of history to download
    "timezone":  "Asia/Karachi", # Timezone for timestamps (any pytz zone)

    # ── Backtest ──────────────────────────────────────────────────────────
    "initial_balance": 100000,   # Starting account balance ($)
    "commission":      0.0002,   # Per-trade commission rate (0.0002 = 0.02%)
    "leverage_margin": 0.00001,  # Margin requirement (0.00001 = very high leverage)

    # ── Metrics ───────────────────────────────────────────────────────────
    "metric_mode": "basic",      # "basic"    → core metrics only (faster)
                                 # "advanced" → adds streaks, direction breakdown,
                                 #              absolute drawdown, best/worst trade
    "min_trades":  20,           # Discard any result with fewer trades than this

    # ── Strategy Parameters ───────────────────────────────────────────────

    # FVG Sensitivity — controls how strict the Fair Value Gap size filter is
    # Options: "Extreme", "High", "Normal", "Low"
    "fvg_sensitivity_values": ["Extreme", "High", "Normal", "Low"],

    # Swing Length — pivot lookback for Order Block detection
    "swing_length_min":  3,
    "swing_length_max":  10,
    "swing_length_step": 1,

    # Require Retracement — wait for price to touch FVG before entering
    "require_retracement_values": [False],   # [False], [True], or [False, True]

    # TP/SL Method
    # "Unicorn"  → SL from rolling 100-bar lowest/highest + ATR offset
    # "Dynamic"  → SL from ATR(50) × risk multiplier
    # "Fixed"    → Fixed % TP and SL (uses tp/sl_percent ranges below)
    "tpsl_methods": ["Unicorn", "Dynamic"],  # add "Fixed" to include Fixed pool

    # use1to1RR — force exactly 1:1 RR for Unicorn and Dynamic methods
    "use_1to1rr_values": [True],             # [True], [False], or [True, False]

    # riskAmount — ATR multiplier for stop-loss distance (Unicorn/Dynamic only)
    # Options: "Highest", "High", "Normal", "Low", "Lowest"
    "risk_amount_values": ["Highest", "High", "Normal", "Low", "Lowest"],

    # Fixed TP % — only used when "Fixed" is in tpsl_methods
    "tp_percent_min":  0.1,
    "tp_percent_max":  1.0,
    "tp_percent_step": 0.1,

    # Fixed SL % — only used when "Fixed" is in tpsl_methods
    "sl_percent_min":  0.1,
    "sl_percent_max":  1.0,
    "sl_percent_step": 0.1,

    # ── CPU ───────────────────────────────────────────────────────────────
    "num_cores": 0,  # 0 = use all available cores on the pod (recommended)
                     # Set to a specific number to cap core usage
}

# ============================================================
#  OUTPUT FILE NAME  (auto-generated from config)
# ============================================================
_ts          = datetime.now().strftime("%Y%m%d_%H%M")
OUTPUT_FILE  = f"{config['symbol']}-{config['interval']}-{config['days_back']}d_{_ts}.csv"

# ============================================================
#  RUN
# ============================================================

if __name__ == '__main__':
    runpod.api_key = RUNPOD_API_KEY

    print("=" * 60)
    print("ICT UNICORN — SUBMITTING JOB TO RUNPOD")
    print("=" * 60)
    print(f"  Symbol          : {config['symbol']}  {config['interval']}  ({config['days_back']}d)")
    print(f"  Timezone        : {config['timezone']}")
    print(f"  Initial Balance : ${config['initial_balance']:,}")
    print(f"  Commission      : {config['commission'] * 100:.4f}%")
    print(f"  Leverage Margin : {config['leverage_margin']}")
    print(f"  Metric Mode     : {config['metric_mode'].upper()}")
    print(f"  Min Trades      : {config['min_trades']}")
    print(f"  TP/SL Methods   : {config['tpsl_methods']}")
    print(f"  FVG Sensitivity : {config['fvg_sensitivity_values']}")
    print(f"  Swing Length    : {config['swing_length_min']}–{config['swing_length_max']} step {config['swing_length_step']}")
    print(f"  Output file     : {OUTPUT_FILE}")
    print("=" * 60)

    endpoint = runpod.Endpoint(ENDPOINT_ID)

    print("\nSubmitting job... (waiting for results, this may take several minutes)")
    result = endpoint.run_sync({"input": config}, timeout=3600)   # 1-hour timeout

    # ── Error handling ──
    if result.get("status") == "error":
        print(f"\nERROR from pod: {result.get('error')}")
        exit(1)

    if "csv_base64" not in result:
        print(f"\nUnexpected response: {result}")
        exit(1)

    # ── Decode and save CSV ──
    csv_bytes = base64.b64decode(result["csv_base64"])
    df        = pd.read_csv(io.BytesIO(csv_bytes))
    df.to_csv(OUTPUT_FILE, index=False)

    # ── Print summary ──
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    cfg = result.get('config_echo', {})
    print(f"  Symbol          : {cfg.get('symbol')}  {cfg.get('interval')}  ({cfg.get('days_back')}d)")
    print(f"  Data range      : {result['data_from']} → {result['data_to']}  ({result['data_bars']:,} bars)")
    print(f"  Initial Balance : ${cfg.get('initial_balance', 0):,}")
    print(f"  Commission      : {(cfg.get('commission', 0) * 100):.4f}%")
    print(f"  Valid results   : {result['total_results']:,} / {result['total_combinations']:,} combinations")
    print(f"  Elapsed         : {result['elapsed_minutes']} min")
    print(f"  Saved to        : {OUTPUT_FILE}")

    print("\n  Best Combination Parameters:")
    for k, v in result['best_params'].items():
        print(f"    {k:22}: {v}")

    print("\n  Best Combination Metrics:")
    for k, v in result['best_metrics'].items():
        print(f"    {k:22}: {v}")

    print("\n  To reproduce best result in unicorn.py:")
    bp = result['best_params']
    print(f"    UnicornStrategy.tpslMethod         = '{bp['tpslMethod']}'")
    print(f"    UnicornStrategy.fvgSensitivity     = '{bp['fvgSensitivity']}'")
    print(f"    UnicornStrategy.swingLength        = {bp['swingLength']}")
    print(f"    UnicornStrategy.requireRetracement = {bp['requireRetracement']}")
    print(f"    UnicornStrategy.use1to1RR          = {bp['use1to1RR']}")
    print(f"    UnicornStrategy.riskAmount         = '{bp['riskAmount']}'")
    print(f"    UnicornStrategy.tpPercent          = {bp['tpPercent']}")
    print(f"    UnicornStrategy.slPercent          = {bp['slPercent']}")
    print("=" * 60)
