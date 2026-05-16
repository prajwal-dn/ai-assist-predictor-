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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_artifacts(cfg: dict, pair: str):
    model_dir = cfg["paths"]["models"]
    model     = joblib.load(os.path.join(model_dir, f"{pair}_xgboost.pkl"))
    scaler    = joblib.load(os.path.join(model_dir, f"{pair}_scaler.pkl"))
    feat_cols = joblib.load(os.path.join(model_dir, f"{pair}_feature_cols.pkl"))
    return model, scaler, feat_cols


def generate_signals(df: pd.DataFrame, model, scaler,
                     feat_cols: list, threshold: float) -> pd.DataFrame:
    """Add probability and signal columns to test dataframe."""
    X     = scaler.transform(df[feat_cols].values)
    proba = model.predict_proba(X)[:, 1]

    df = df.copy()
    df["prob_up"]   = proba
    df["prob_down"] = 1 - proba
    df["confidence"] = df[["prob_up", "prob_down"]].max(axis=1)

    df["signal"] = np.where(
        df["prob_up"]   >= threshold, 1,   # BUY
        np.where(
        df["prob_down"] >= threshold, -1,  # SELL
        0                                  # FLAT
    ))
    return df


def simulate_trades(df: pd.DataFrame, cfg: dict, horizon: int) -> pd.DataFrame:
    """
    Simulate trade P&L.
    Entry: on signal candle close.
    Exit:  `horizon` candles later.
    Risk:  ATR-based stop, max 1% account per trade.
    """
    atr_mult   = cfg["risk"]["atr_stop_multiplier"]
    risk_pct   = cfg["risk"]["max_risk_per_trade"]
    account    = 10_000.0  # hypothetical starting balance in USD

    trades = []

    for i, row in df.iterrows():
        if row["signal"] == 0:
            continue
        if i + horizon >= len(df):
            break

        entry_price  = row["close"]
        exit_price   = df.iloc[i + horizon]["close"]
        atr          = row["atr"]
        stop_size    = atr * atr_mult
        direction    = row["signal"]   # 1=long, -1=short

        # Position size based on risk
        pip_risk     = stop_size
        if pip_risk == 0:
            continue
        position_size = (account * risk_pct) / pip_risk

        # P&L
        raw_pnl = direction * (exit_price - entry_price) * position_size
        pnl_pct = direction * (exit_price - entry_price) / entry_price

        trades.append({
            "entry_time":  row["datetime"],
            "exit_time":   df.iloc[i + horizon]["datetime"],
            "direction":   "BUY" if direction == 1 else "SELL",
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "confidence":  row["confidence"],
            "atr":         atr,
            "pnl_pct":     pnl_pct,
            "pnl_usd":     raw_pnl,
            "win":         1 if raw_pnl > 0 else 0,
        })

    return pd.DataFrame(trades)


def print_summary(trades_df: pd.DataFrame):
    if trades_df.empty:
        print("[WARN] No trades were generated. Try lowering confidence_threshold.")
        return

    total_trades = len(trades_df)
    wins         = trades_df["win"].sum()
    losses       = total_trades - wins
    win_rate     = wins / total_trades * 100
    avg_win      = trades_df[trades_df["win"] == 1]["pnl_pct"].mean() * 100
    avg_loss     = trades_df[trades_df["win"] == 0]["pnl_pct"].mean() * 100
    total_pnl    = trades_df["pnl_usd"].sum()
    max_dd       = compute_max_drawdown(trades_df["pnl_usd"].cumsum())
    profit_factor = (
        trades_df[trades_df["win"] == 1]["pnl_usd"].sum() /
        abs(trades_df[trades_df["win"] == 0]["pnl_usd"].sum() + 1e-10)
    )

    print("\n─── Backtest Summary ─────────────────────────────────────────")
    print(f"  Total trades     : {total_trades}")
    print(f"  Wins / Losses    : {wins} / {losses}")
    print(f"  Win rate         : {win_rate:.1f}%")
    print(f"  Avg win          : {avg_win:.3f}%")
    print(f"  Avg loss         : {avg_loss:.3f}%")
    print(f"  Profit factor    : {profit_factor:.2f}   (>1.5 is good)")
    print(f"  Total P&L (USD)  : ${total_pnl:,.2f}   (on $10,000 account)")
    print(f"  Max drawdown     : {max_dd:.1f}%")
    print("──────────────────────────────────────────────────────────────\n")


def compute_max_drawdown(cumulative_pnl: pd.Series) -> float:
    """Max peak-to-trough drawdown as a percentage of peak equity."""
    equity   = 10_000 + cumulative_pnl
    peak     = equity.cummax()
    drawdown = (equity - peak) / peak * 100
    return drawdown.min()


def plot_results(df: pd.DataFrame, trades_df: pd.DataFrame, pair: str):
    """Save equity curve and signal chart to models/ folder."""
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
        equity = 10_000 + trades_df["pnl_usd"].cumsum()
        ax3.plot(pd.to_datetime(trades_df["entry_time"]), equity,
                 color="#e67e22", linewidth=1.2)
        ax3.axhline(y=10_000, color="gray", linestyle="--", linewidth=0.8)
    ax3.set_ylabel("Equity ($)")
    ax3.set_xlabel("Date")

    plt.tight_layout()
    out_path = f"models/{pair}_backtest.png"
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
    test_file = os.path.join(
        cfg["paths"]["labelled_data"], f"{pair}_{timeframe}_test.csv"
    )
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test data not found: {test_file}. Run labeller.py first.")

    print(f"[INFO] Loading test data from {test_file}")
    test_df = pd.read_csv(test_file, parse_dates=["datetime"])
    print(f"[INFO] {len(test_df):,} rows in test set.")

    model, scaler, feat_cols = load_artifacts(cfg, pair)

    test_df = generate_signals(test_df, model, scaler, feat_cols, threshold)
    trades  = simulate_trades(test_df, cfg, horizon)

    print_summary(trades)
    plot_results(test_df, trades, pair)

    # Save trades CSV
    trades_file = os.path.join(cfg["paths"]["models"], f"{pair}_trades.csv")
    trades.to_csv(trades_file, index=False)
    print(f"[INFO] Trade log saved → {trades_file}")
    print("[DONE] backtest complete.")


if __name__ == "__main__":
    main()
