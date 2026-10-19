"""
backtest.py
-----------
Replays the trained model's signals on historical test data
and computes trading performance metrics.

Usage:
    python src/backtest.py
"""

import os
import yaml
import joblib
import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(BASE_DIR, "config.yaml")
    elif not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid application configuration: {path}")
    return cfg


def load_artifacts(cfg: dict, pair: str):
    model_dir = cfg["paths"]["models"]
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(BASE_DIR, model_dir)
    artifacts = [
        os.path.join(model_dir, f"{pair}_xgboost.pkl"),
        os.path.join(model_dir, f"{pair}_scaler.pkl"),
        os.path.join(model_dir, f"{pair}_feature_cols.pkl"),
    ]
    for path in artifacts:
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            raise FileNotFoundError(f"Missing/empty backtest model artifact: {path}")
        with open(path, "rb") as handle:
            if handle.read(256).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"Model artifact is only a Git LFS pointer: {path}")
    model, scaler, feat_cols = (joblib.load(path) for path in artifacts)
    metadata_path = os.path.join(model_dir, f"{pair}_metadata.json")
    if os.path.isfile(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("timeframe") and metadata["timeframe"] != cfg["timeframe"]:
            raise ValueError("Model timeframe metadata does not match config.yaml.")
        if metadata.get("feature_columns") and list(metadata["feature_columns"]) != list(feat_cols):
            raise ValueError("Model metadata feature list does not match the feature artifact.")
    n_features = getattr(scaler, "n_features_in_", None)
    if n_features is not None and int(n_features) != len(feat_cols):
        raise ValueError("The scaler feature count does not match the feature artifact.")
    return model, scaler, list(feat_cols)


def generate_signals(df: pd.DataFrame, model, scaler,
                     feat_cols: list, threshold: float) -> pd.DataFrame:
    """Add validated probabilities and BUY/SELL/FLAT signals to held-out data."""
    if not 0.5 < float(threshold) < 1.0:
        raise ValueError("confidence_threshold must be greater than 0.5 and less than 1.")
    missing = [column for column in feat_cols if column not in df.columns]
    if missing:
        raise ValueError(f"Test data is missing model features: {missing}")
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("Test features contain missing or non-finite values.")
    X_scaled = scaler.transform(X)
    probabilities = np.asarray(model.predict_proba(X_scaled), dtype=float)
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    up_index = np.flatnonzero(classes == 1)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(df) or len(up_index) != 1:
        raise ValueError("Model probabilities must include exactly one class labelled 1 (UP).")
    proba_up = probabilities[:, int(up_index[0])]
    if not np.isfinite(proba_up).all() or ((proba_up < 0) | (proba_up > 1)).any():
        raise ValueError("Model produced invalid probabilities.")
    result = df.copy()
    result["prob_up"] = proba_up
    result["prob_down"] = 1.0 - proba_up
    result["confidence"] = result[["prob_up", "prob_down"]].max(axis=1)
    result["signal"] = np.where(
        result["prob_up"] >= threshold, 1,
        np.where(result["prob_down"] >= threshold, -1, 0)
    )
    return result


def quote_currency(pair: str) -> str:
    """Return the quote currency; backtest equity is denominated in this currency."""
    code = pair.upper().replace("=X", "")
    if "-" in code:
        parts = code.split("-", 1)
        return parts[1] if len(parts) == 2 and parts[1] else "QUOTE"
    if len(code) == 6 and code.isalpha():
        return code[3:]
    return "QUOTE"


def simulate_trades(
    df: pd.DataFrame, cfg: dict, horizon: int, starting_balance: float = 10_000.0
) -> pd.DataFrame:
    """Simulate one position at a time, applying ATR stop/target checks on each candle.

    Entry is at the signal candle close. Stop/target levels are inspected on each
    following candle through the horizon. If both trigger within the same candle,
    the stop is assumed first (conservative). Gap-through stops fill at the candle
    open. Fees, spread and slippage other than stop gaps are not modeled.
    """
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Backtest horizon must be a positive integer number of candles.")
    risk = cfg.get("risk", {})
    atr_mult = float(risk.get("atr_stop_multiplier", 1.5))
    risk_pct = float(risk.get("max_risk_per_trade", 0.01))
    if not math.isfinite(atr_mult) or atr_mult <= 0:
        raise ValueError("atr_stop_multiplier must be finite and positive.")
    if not math.isfinite(risk_pct) or not 0 < risk_pct <= 1:
        raise ValueError("max_risk_per_trade must be in (0, 1].")
    if not math.isfinite(starting_balance) or starting_balance <= 0:
        raise ValueError("Starting balance must be finite and positive.")
    required = {"open", "high", "low", "close", "atr", "signal", "datetime"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Backtest data is missing columns: {sorted(missing)}")
    if df.empty:
        return pd.DataFrame()
    candles = df.copy().reset_index(drop=True)
    for column in ("open", "high", "low", "close", "atr", "signal"):
        candles[column] = pd.to_numeric(candles[column], errors="coerce")
    price_values = candles[["open", "high", "low", "close", "atr"]].to_numpy(dtype=float)
    if not np.isfinite(price_values).all():
        raise ValueError("Backtest data contains non-finite OHLC/ATR values.")
    candles = candles.sort_values("datetime").reset_index(drop=True)
    equity = float(starting_balance)
    trades = []
    last_exit_index = -1
    confidence_series = candles["confidence"] if "confidence" in candles.columns else pd.Series(np.nan, index=candles.index)

    for i, row in candles.iterrows():
        if i <= last_exit_index:
            continue
        direction = int(row["signal"])
        if direction not in (-1, 1):
            continue
        if i + horizon >= len(candles):
            break
        entry = float(row["close"])
        atr = float(row["atr"])
        stop_size = atr * atr_mult
        if entry <= 0 or atr <= 0 or not math.isfinite(stop_size) or stop_size <= 0:
            continue
        stop_price = entry - stop_size if direction == 1 else entry + stop_size
        target_price = entry + 2 * stop_size if direction == 1 else entry - 2 * stop_size
        risk_cash = equity * risk_pct
        units = risk_cash / stop_size
        exit_price = float(candles.iloc[i + horizon]["close"])
        exit_index = i + horizon
        exit_reason = "HORIZON"
        stop_triggered = False

        for j in range(i + 1, i + horizon + 1):
            candle = candles.iloc[j]
            op, hi, lo = float(candle["open"]), float(candle["high"]), float(candle["low"])
            if direction == 1:
                gap_stop = op <= stop_price
                gap_target = op >= target_price
                hit_stop = lo <= stop_price
                hit_target = hi >= target_price
            else:
                gap_stop = op >= stop_price
                gap_target = op <= target_price
                hit_stop = hi >= stop_price
                hit_target = lo <= target_price

            # If the market gaps through the stop, use the worse open fill.
            if gap_stop:
                exit_price, exit_index, exit_reason = op, j, "STOP_LOSS_GAP"
                stop_triggered = True
                break
            # When stop and target are both reachable inside one candle, assume
            # the less favorable order because OHLC candles reveal no path order.
            if hit_stop and hit_target:
                exit_price, exit_index, exit_reason = stop_price, j, "STOP_LOSS_AMBIGUOUS"
                stop_triggered = True
                break
            if hit_stop:
                exit_price, exit_index, exit_reason = stop_price, j, "STOP_LOSS"
                stop_triggered = True
                break
            if gap_target or hit_target:
                exit_price, exit_index, exit_reason = target_price, j, "TAKE_PROFIT"
                break

        pnl = direction * (exit_price - entry) * units
        equity_before = equity
        equity = max(0.0, equity + pnl)
        trade = {
            "entry_time": row["datetime"],
            "exit_time": candles.iloc[exit_index]["datetime"],
            "direction": "BUY" if direction == 1 else "SELL",
            "entry_price": entry,
            "exit_price": exit_price,
            "stop_loss": stop_price,
            "take_profit": target_price,
            "exit_reason": exit_reason,
            "confidence": float(confidence_series.iloc[i]) if pd.notna(confidence_series.iloc[i]) else None,
            "atr": atr,
            "position_units": units,
            "risk_amount": risk_cash,
            "equity_before": equity_before,
            "equity_after": equity,
            "pnl_quote": pnl,
            "pnl_pct": (pnl / equity_before) * 100.0,
            "win": int(pnl > 0),
        }
        trades.append(trade)
        last_exit_index = exit_index
        if equity <= 0:
            break
    return pd.DataFrame(trades)


def print_summary(trades_df: pd.DataFrame, currency: str = "QUOTE",
                  starting_balance: float = 10_000.0):
    if trades_df.empty:
        print("[WARN] No trades were generated; there is no performance summary.")
        return

    total_trades = len(trades_df)
    wins = int(trades_df["win"].sum())
    losses = total_trades - wins
    win_rate = wins / total_trades * 100
    avg_win = trades_df.loc[trades_df["win"] == 1, "pnl_pct"].mean()
    avg_loss = trades_df.loc[trades_df["win"] == 0, "pnl_pct"].mean()
    total_pnl = trades_df["pnl_quote"].sum()
    max_dd = compute_max_drawdown(trades_df["pnl_quote"].cumsum(), starting_balance)
    gross_profit = trades_df.loc[trades_df["pnl_quote"] > 0, "pnl_quote"].sum()
    gross_loss = abs(trades_df.loc[trades_df["pnl_quote"] < 0, "pnl_quote"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    print("\n--- Backtest Summary (quote currency; costs not modeled) ---")
    print(f"  Total trades       : {total_trades}")
    print(f"  Wins / Losses      : {wins} / {losses}")
    print(f"  Win rate           : {win_rate:.1f}%")
    print(f"  Average win        : {avg_win:.3f}%" if pd.notna(avg_win) else "  Average win        : n/a")
    print(f"  Average loss       : {avg_loss:.3f}%" if pd.notna(avg_loss) else "  Average loss       : n/a")
    print(f"  Gross profit       : {currency} {gross_profit:,.2f}")
    print(f"  Gross loss         : {currency} {gross_loss:,.2f}")
    print(f"  Net P&L            : {currency} {total_pnl:,.2f}")
    print(f"  Ending equity      : {currency} {starting_balance + total_pnl:,.2f}")
    print(f"  Profit factor      : {profit_factor:.2f}")
    print(f"  Max drawdown       : {max_dd:.1f}%")
    print("-----------------------------------------------------------------\n")


def compute_max_drawdown(cumulative_pnl: pd.Series, starting_balance: float = 10_000.0) -> float:
    """Max peak-to-trough drawdown as a percentage of peak equity."""
    equity   = starting_balance + cumulative_pnl
    peak     = equity.cummax()
    drawdown = (equity - peak) / peak * 100
    return drawdown.min()


def plot_results(
    df: pd.DataFrame, trades_df: pd.DataFrame, pair: str,
    starting_balance: float = 10_000.0, currency: str = "QUOTE",
    output_dir: str = None,
):
    """Save a chart using the same P&L currency and starting balance as simulation."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Price + signals
    ax1 = axes[0]
    ax1.plot(pd.to_datetime(df["datetime"]), df["close"], color="#4a90d9",
             linewidth=0.8, label="Close")
    if not trades_df.empty:
        buys  = trades_df[trades_df["direction"] == "BUY"]
        sells = trades_df[trades_df["direction"] == "SELL"]
        ax1.scatter(pd.to_datetime(buys["entry_time"]),
                    buys["entry_price"], marker="^", color="#2ecc71",
                    s=60, zorder=5, label="BUY signal")
        ax1.scatter(pd.to_datetime(sells["entry_time"]),
                    sells["entry_price"], marker="v", color="#e74c3c",
                    s=60, zorder=5, label="SELL signal")
    ax1.set_ylabel("Price")
    ax1.legend(fontsize=8)
    ax1.set_title(f"{pair.upper()} — Backtest Signals")

    # Confidence
    ax2 = axes[1]
    ax2.fill_between(pd.to_datetime(df["datetime"]), df["confidence"],
                     0.5, alpha=0.4, color="#9b59b6")
    ax2.axhline(y=df["confidence"].mean(), color="gray",
                linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Confidence")
    ax2.set_ylim(0, 1)

    # Equity curve
    ax3 = axes[2]
    if not trades_df.empty:
        equity = starting_balance + trades_df["pnl_quote"].cumsum()
        ax3.plot(pd.to_datetime(trades_df["exit_time"]), equity,
                 color="#e67e22", linewidth=1.2)
        ax3.axhline(y=starting_balance, color="gray", linestyle="--", linewidth=0.8)
    ax3.set_ylabel(f"Equity ({currency})")
    ax3.set_xlabel("Date")

    plt.tight_layout()
    output_dir = output_dir or os.path.join(BASE_DIR, "models")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{pair}_backtest.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"[INFO] Chart saved → {out_path}")
    plt.close()


def main():
    cfg       = load_config()
    pair      = cfg["pair"].replace("=X", "").lower()
    timeframe = cfg["timeframe"]
    horizon   = cfg["forecast_horizon"]
    threshold = cfg["confidence_threshold"]

    # Load test data (never seen during training)
    label_dir = cfg["paths"]["labelled_data"]
    if not os.path.isabs(label_dir):
        label_dir = os.path.join(BASE_DIR, label_dir)
    test_file = os.path.join(label_dir, f"{pair}_{timeframe}_test.csv")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test data not found: {test_file}. Run labeller.py first.")

    print(f"[INFO] Loading test data from {test_file}")
    test_df = pd.read_csv(test_file, parse_dates=["datetime"])
    print(f"[INFO] {len(test_df):,} rows in test set.")

    model, scaler, feat_cols = load_artifacts(cfg, pair)

    test_df = generate_signals(test_df, model, scaler, feat_cols, threshold)
    starting_balance = float(cfg.get("backtest", {}).get("starting_balance", 10_000.0))
    currency = quote_currency(cfg["pair"])
    trades  = simulate_trades(test_df, cfg, horizon, starting_balance=starting_balance)

    print_summary(trades, currency=currency, starting_balance=starting_balance)

    # All backtest outputs honor the configured model directory.
    model_dir = cfg["paths"]["models"]
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(BASE_DIR, model_dir)
    os.makedirs(model_dir, exist_ok=True)
    plot_results(test_df, trades, pair, starting_balance=starting_balance,
                 currency=currency, output_dir=model_dir)

    # Save trades CSV
    trades_file = os.path.join(model_dir, f"{pair}_trades.csv")
    trades.to_csv(trades_file, index=False)
    print(f"[INFO] Trade log saved → {trades_file}")
    print("[DONE] backtest complete.")


if __name__ == "__main__":
    main()
