"""
ICT UNICORN STRATEGY OPTIMIZER — RunPod Core Module
All optimization logic as pure functions. Entry point: run_optimization(config).
No interactive CLI, no file I/O — data comes from Binance Vision, results returned as base64 CSV.
"""

from backtesting.lib import FractionalBacktest as Backtest
import pandas as pd
import numpy as np
from unicorn import UnicornStrategy
from binance_vision import download_from_vision
import warnings
import base64
import io
from multiprocessing import Pool, cpu_count
from datetime import datetime

warnings.filterwarnings('ignore')


# ── Module-level globals — set in each subprocess by init_worker ──
global_df              = None
global_metric_mode     = 'basic'
global_initial_balance = 100000
global_commission      = 0.0002
global_margin          = 0.00001


# =========================================================================
# WORKER INITIALIZER
# =========================================================================

def init_worker(df_data, metric_mode, initial_balance, commission, margin):
    """Initialize each worker process with shared read-only globals."""
    global global_df, global_metric_mode, global_initial_balance
    global global_commission, global_margin
    global_df              = df_data
    global_metric_mode     = metric_mode
    global_initial_balance = initial_balance
    global_commission      = commission
    global_margin          = margin


# =========================================================================
# METRIC HELPER FUNCTIONS  (identical to optimize.py)
# =========================================================================

def calculate_absolute_drawdown(equity_df, initial_balance):
    """Drawdown from initial balance (not rolling peak — absolute floor view)."""
    if equity_df is None or len(equity_df) == 0:
        return None
    try:
        min_equity  = equity_df['Equity'].min()
        absolute_dd = ((initial_balance - min_equity) / initial_balance) * 100
        return -abs(absolute_dd)
    except Exception:
        return None


def calculate_streak_metrics(trades_df):
    """Compute maximum consecutive winning and losing streaks."""
    if trades_df is None or len(trades_df) == 0:
        return None, None
    returns          = trades_df['ReturnPct'] if 'ReturnPct' in trades_df.columns else trades_df['PnL']
    is_win           = returns > 0
    max_win_streak   = max_loss_streak = 0
    curr_win_streak  = curr_loss_streak = 0
    for win in is_win:
        if win:
            curr_win_streak  += 1
            curr_loss_streak  = 0
            max_win_streak    = max(max_win_streak,  curr_win_streak)
        else:
            curr_loss_streak += 1
            curr_win_streak   = 0
            max_loss_streak   = max(max_loss_streak, curr_loss_streak)
    return max_win_streak, max_loss_streak


def calculate_avg_win_loss_ratio(trades_df):
    """Average winning trade / absolute average losing trade."""
    if trades_df is None or len(trades_df) == 0:
        return None
    returns  = trades_df['ReturnPct'] if 'ReturnPct' in trades_df.columns else trades_df['PnL']
    winning  = returns[returns > 0]
    losing   = returns[returns < 0]
    if len(winning) == 0 or len(losing) == 0:
        return None
    avg_loss = abs(losing.mean())
    return winning.mean() / avg_loss if avg_loss != 0 else None


def calculate_winning_losing_counts(trades_df):
    """Raw count of winning and losing trades."""
    if trades_df is None or len(trades_df) == 0:
        return 0, 0
    returns = trades_df['ReturnPct'] if 'ReturnPct' in trades_df.columns else trades_df['PnL']
    return len(returns[returns > 0]), len(returns[returns < 0])


def calculate_direction_metrics(trades_df):
    """Long vs Short win rate (%) and total PnL."""
    if trades_df is None or len(trades_df) == 0:
        return None, None, None, None
    long_trades  = trades_df[trades_df['Size'] > 0]
    short_trades = trades_df[trades_df['Size'] < 0]
    long_wr = long_pnl = short_wr = short_pnl = None
    if len(long_trades) > 0:
        long_wr  = (long_trades['PnL']  > 0).sum() / len(long_trades)  * 100
        long_pnl = long_trades['PnL'].sum()
    if len(short_trades) > 0:
        short_wr  = (short_trades['PnL'] > 0).sum() / len(short_trades) * 100
        short_pnl = short_trades['PnL'].sum()
    return long_wr, long_pnl, short_wr, short_pnl


# =========================================================================
# WORKER FUNCTION  (identical logic to optimize.py run_single_backtest)
# =========================================================================

def run_single_backtest(args):
    """
    Worker — executes one parameter combination and returns a result dict.
    Parameter dict keys must match UnicornStrategy class attribute names exactly.
    """
    combination_num, params, min_trades = args
    try:
        global global_df, global_metric_mode, global_initial_balance
        global global_commission, global_margin

        bt_worker = Backtest(
            global_df,
            UnicornStrategy,
            cash=global_initial_balance,
            commission=global_commission,
            exclusive_orders=True,
            trade_on_close=True,
            margin=global_margin
        )
        stats = bt_worker.run(**params)

        # Hard gate — discard statistically unreliable results.
        if stats['# Trades'] < min_trades:
            return None

        result = {'Combination': combination_num, **params}

        # ── BASIC METRICS (always collected) ──
        result.update({
            'Return [%]':          stats['Return [%]'],
            'Sharpe Ratio':        stats['Sharpe Ratio'],
            'Sortino Ratio':       stats.get('Sortino Ratio',  None),
            'Calmar Ratio':        stats.get('Calmar Ratio',   None),
            'Max. Drawdown [%]':   stats['Max. Drawdown [%]'],
            '# Trades':            stats['# Trades'],
            'Win Rate [%]':        stats['Win Rate [%]'],
            'Profit Factor':       stats['Profit Factor'],
            'Expectancy [%]':      stats.get('Expectancy [%]', None),
            'Avg. Trade [%]':      stats['Avg. Trade [%]'],
            'Avg. Trade Duration': str(stats['Avg. Trade Duration']),
            'Exposure Time [%]':   stats['Exposure Time [%]'],
        })

        # ── ADVANCED METRICS (optional) ──
        if global_metric_mode == 'advanced':
            trades_df = stats.get('_trades',       None)
            equity_df = stats.get('_equity_curve', None)

            max_win_streak, max_loss_streak = calculate_streak_metrics(trades_df)
            avg_win_loss                    = calculate_avg_win_loss_ratio(trades_df)
            winning_cnt, losing_cnt         = calculate_winning_losing_counts(trades_df)
            long_wr, long_pnl, short_wr, short_pnl = calculate_direction_metrics(trades_df)
            absolute_dd                     = calculate_absolute_drawdown(equity_df, global_initial_balance)

            result.update({
                'Max. Absolute DD [%]': absolute_dd,
                'Avg Win / Avg Loss':   avg_win_loss,
                'Max Loss Streak':      max_loss_streak,
                'Max Win Streak':       max_win_streak,
                'Winning Trades':       winning_cnt,
                'Losing Trades':        losing_cnt,
                'Best Trade [%]':       stats.get('Best Trade [%]',  None),
                'Worst Trade [%]':      stats.get('Worst Trade [%]', None),
                'Max. Trade Duration':  str(stats['Max. Trade Duration']),
                'Long Win Rate [%]':    long_wr,
                'Long Total PnL':       long_pnl,
                'Short Win Rate [%]':   short_wr,
                'Short Total PnL':      short_pnl,
            })

        return result

    except Exception as e:
        return {
            'Combination':  combination_num,
            **params,
            'Return [%]':   None,
            'Sharpe Ratio': None,
            'Error':        str(e),
        }


# =========================================================================
# COMBINATION GENERATOR  (identical logic to optimize.py)
# =========================================================================

def build_combinations(
    fvg_sensitivity_values,
    swing_length_values,
    require_retracement_values,
    tpsl_methods,
    use_1to1rr_values,
    risk_amount_values,
    tp_percent_values,
    sl_percent_values,
    min_trades,
):
    """
    Generate all valid parameter dicts, separated by tpslMethod pool.

    Pool A — "Unicorn" and "Dynamic":
        fvgSensitivity × swingLength × requireRetracement × use1to1RR × riskAmount

    Pool B — "Fixed":
        fvgSensitivity × swingLength × requireRetracement × tpPercent × slPercent
    """
    worker_args = []
    combo_num   = 0

    # ── Pool A: Unicorn + Dynamic ──
    methods_ab = [m for m in tpsl_methods if m in ("Unicorn", "Dynamic")]
    for method in methods_ab:
        for fvg_sens in fvg_sensitivity_values:
            for swing in swing_length_values:
                for req_ret in require_retracement_values:
                    for use_rr in use_1to1rr_values:
                        for risk_amt in risk_amount_values:
                            combo_num += 1
                            params = {
                                'tpslMethod':          method,
                                'fvgSensitivity':      fvg_sens,
                                'swingLength':         swing,
                                'requireRetracement':  req_ret,
                                'use1to1RR':           use_rr,
                                'riskAmount':          risk_amt,
                                # Fixed-mode params kept at strategy defaults
                                'tpPercent':           0.3,
                                'slPercent':           0.4,
                            }
                            worker_args.append((combo_num, params, min_trades))

    # ── Pool B: Fixed ──
    if "Fixed" in tpsl_methods:
        for fvg_sens in fvg_sensitivity_values:
            for swing in swing_length_values:
                for req_ret in require_retracement_values:
                    for tp_pct in tp_percent_values:
                        for sl_pct in sl_percent_values:
                            combo_num += 1
                            params = {
                                'tpslMethod':          'Fixed',
                                'fvgSensitivity':      fvg_sens,
                                'swingLength':         swing,
                                'requireRetracement':  req_ret,
                                # Irrelevant for Fixed but must be present
                                'use1to1RR':           True,
                                'riskAmount':          'Normal',
                                'tpPercent':           round(tp_pct, 4),
                                'slPercent':           round(sl_pct, 4),
                            }
                            worker_args.append((combo_num, params, min_trades))

    return worker_args, combo_num


# =========================================================================
# MAIN ENTRY POINT
# =========================================================================

def run_optimization(config: dict) -> dict:
    """
    Called by handler.py. Downloads fresh data from Binance Vision,
    runs the full parameter optimization, and returns results as base64 CSV.

    config keys: see run_local.py for the full config reference.
    """

    # ── Data config ──
    symbol      = config.get('symbol',    'ETHUSDT.P')
    interval    = config.get('interval',  '3m')
    days_back   = config.get('days_back', 365)
    timezone    = config.get('timezone',  'UTC')

    # ── Backtest config ──
    initial_balance = config.get('initial_balance', 100000)
    commission      = config.get('commission',      0.0002)
    leverage_margin = config.get('leverage_margin', 0.00001)

    # ── Run config ──
    metric_mode = config.get('metric_mode', 'basic')
    num_cores   = config.get('num_cores',   0)
    min_trades  = config.get('min_trades',  20)

    # ── Parameter config ──
    fvg_sensitivity_values     = config.get('fvg_sensitivity_values',     ["Extreme", "High", "Normal", "Low"])
    swing_length_min           = config.get('swing_length_min',           3)
    swing_length_max           = config.get('swing_length_max',           10)
    swing_length_step          = config.get('swing_length_step',          1)
    require_retracement_values = config.get('require_retracement_values', [False])
    tpsl_methods               = config.get('tpsl_methods',               ["Unicorn", "Dynamic"])
    use_1to1rr_values          = config.get('use_1to1rr_values',          [True])
    risk_amount_values         = config.get('risk_amount_values',         ["Highest", "High", "Normal", "Low", "Lowest"])
    tp_percent_min             = config.get('tp_percent_min',             0.1)
    tp_percent_max             = config.get('tp_percent_max',             1.0)
    tp_percent_step            = config.get('tp_percent_step',            0.1)
    sl_percent_min             = config.get('sl_percent_min',             0.1)
    sl_percent_max             = config.get('sl_percent_max',             1.0)
    sl_percent_step            = config.get('sl_percent_step',            0.1)

    swing_length_values = list(range(swing_length_min, swing_length_max + 1, swing_length_step))
    tp_percent_values   = [round(v, 4) for v in np.arange(tp_percent_min, tp_percent_max + 1e-9, tp_percent_step)]
    sl_percent_values   = [round(v, 4) for v in np.arange(sl_percent_min, sl_percent_max + 1e-9, sl_percent_step)]

    # ── Download data from Binance Vision ──
    print(f"\nDownloading {symbol} {interval} ({days_back}d)...")
    df = download_from_vision(
        symbol=symbol,
        interval=interval,
        days_back=days_back,
        timezone=timezone,
    )
    if df is None or len(df) == 0:
        raise RuntimeError(f"Failed to download data for {symbol} {interval}")

    df.rename(columns={
        'open':  'Open',
        'high':  'High',
        'low':   'Low',
        'close': 'Close',
        'volume': 'Volume',
    }, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    data_bars = len(df)
    data_from = str(df.index[0].date())
    data_to   = str(df.index[-1].date())
    print(f"Data loaded: {data_bars:,} bars  |  {data_from} → {data_to}")

    # ── Build combinations ──
    worker_args, total_combinations = build_combinations(
        fvg_sensitivity_values,
        swing_length_values,
        require_retracement_values,
        tpsl_methods,
        use_1to1rr_values,
        risk_amount_values,
        tp_percent_values,
        sl_percent_values,
        min_trades,
    )

    # ── Pool size breakdown ──
    methods_ab = [m for m in tpsl_methods if m in ("Unicorn", "Dynamic")]
    pool_ab = (
        len(methods_ab) * len(fvg_sensitivity_values) * len(swing_length_values) *
        len(require_retracement_values) * len(use_1to1rr_values) * len(risk_amount_values)
    )
    pool_fixed = 0
    if "Fixed" in tpsl_methods:
        pool_fixed = (
            len(fvg_sensitivity_values) * len(swing_length_values) *
            len(require_retracement_values) * len(tp_percent_values) * len(sl_percent_values)
        )

    print(f"\nPool A (Unicorn+Dynamic) : {pool_ab:,}")
    print(f"Pool B (Fixed)           : {pool_fixed:,}")
    print(f"Total combinations       : {total_combinations:,}")
    print(f"Metric mode              : {metric_mode.upper()}")
    print(f"Min trades filter        : {min_trades}")
    print(f"Initial balance          : ${initial_balance:,}")
    print(f"Commission               : {commission * 100:.4f}%")
    print(f"Leverage margin          : {leverage_margin}")

    # ── CPU cores ──
    available = cpu_count()
    cores     = available if num_cores == 0 else min(num_cores, available)
    print(f"Using {cores} CPU cores (of {available} available)\n")

    # ── Run optimization ──
    print(f"Starting optimization... ({len(worker_args):,} combinations)")
    print("=" * 60)

    start_time   = datetime.now()
    results_list = []

    with Pool(
        processes=cores,
        initializer=init_worker,
        initargs=(df, metric_mode, initial_balance, commission, leverage_margin),
    ) as pool:
        for i, result in enumerate(pool.imap_unordered(run_single_backtest, worker_args), 1):
            if result is not None:
                results_list.append(result)
            if i % 50 == 0 or i == len(worker_args):
                elapsed = (datetime.now() - start_time).total_seconds()
                rate    = i / elapsed if elapsed > 0 else 0
                eta_s   = (len(worker_args) - i) / rate if rate > 0 else 0
                print(
                    f"  [{i:>6,}/{len(worker_args):,}]  "
                    f"{rate:5.1f} combos/s  "
                    f"ETA: {int(eta_s // 60):02d}m {int(eta_s % 60):02d}s  "
                    f"Valid results: {len(results_list):,}"
                )

    elapsed_total = (datetime.now() - start_time).total_seconds()
    print(f"\nOptimization done in {elapsed_total / 60:.1f} minutes")

    # ── Process results ──
    if not results_list:
        raise RuntimeError(
            f"No valid results — all combinations filtered out "
            f"(min_trades={min_trades}). Try lowering min_trades."
        )

    all_results_df = pd.DataFrame(results_list)

    # Fill NaN/None in numeric columns
    numeric_cols = all_results_df.select_dtypes(include=[np.number]).columns
    all_results_df[numeric_cols] = all_results_df[numeric_cols].fillna(0)
    nullable_metric_cols = [
        'Sortino Ratio', 'Calmar Ratio', 'Expectancy [%]',
        'Best Trade [%]', 'Worst Trade [%]',
        'Max. Absolute DD [%]', 'Avg Win / Avg Loss',
        'Max Win Streak', 'Max Loss Streak',
        'Long Win Rate [%]', 'Long Total PnL',
        'Short Win Rate [%]', 'Short Total PnL',
        'Winning Trades', 'Losing Trades',
    ]
    for col in nullable_metric_cols:
        if col in all_results_df.columns:
            all_results_df[col] = pd.to_numeric(all_results_df[col], errors='coerce').fillna(0)

    # Sort by Sharpe Ratio
    all_results_df = all_results_df.sort_values('Sharpe Ratio', ascending=False)
    all_results_df.insert(0, 'Rank', range(1, len(all_results_df) + 1))

    # ── Encode results as base64 CSV ──
    buf     = io.BytesIO()
    all_results_df.to_csv(buf, index=False)
    csv_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # ── Build response ──
    best = all_results_df.iloc[0]

    return {
        "status":            "success",
        "csv_base64":        csv_b64,
        "total_results":     len(all_results_df),
        "total_combinations": total_combinations,
        "elapsed_minutes":   round(elapsed_total / 60, 2),
        "data_bars":         data_bars,
        "data_from":         data_from,
        "data_to":           data_to,
        "config_echo": {
            "symbol":          symbol,
            "interval":        interval,
            "days_back":       days_back,
            "initial_balance": initial_balance,
            "commission":      commission,
            "leverage_margin": leverage_margin,
            "metric_mode":     metric_mode,
            "min_trades":      min_trades,
        },
        "best_params": {
            "tpslMethod":         str(best.get('tpslMethod',         'N/A')),
            "fvgSensitivity":     str(best.get('fvgSensitivity',     'N/A')),
            "swingLength":        int(best.get('swingLength',         10)),
            "requireRetracement": bool(best.get('requireRetracement', False)),
            "use1to1RR":          bool(best.get('use1to1RR',          True)),
            "riskAmount":         str(best.get('riskAmount',          'Normal')),
            "tpPercent":          float(best.get('tpPercent',         0.3)),
            "slPercent":          float(best.get('slPercent',         0.4)),
        },
        "best_metrics": {
            "Sharpe Ratio":      round(float(best.get('Sharpe Ratio')     or 0), 4),
            "Return [%]":        round(float(best.get('Return [%]')       or 0), 4),
            "Max. Drawdown [%]": round(float(best.get('Max. Drawdown [%]') or 0), 4),
            "Win Rate [%]":      round(float(best.get('Win Rate [%]')     or 0), 4),
            "Profit Factor":     round(float(best.get('Profit Factor')    or 0), 4),
            "# Trades":          int(best.get('# Trades') or 0),
        },
    }
