import os
from pathlib import Path
import polars as pl
import numpy as np
import duckdb

rootPath = Path(__file__).resolve().parents[1]
MC_DUCKDB = Path(f"{rootPath}/data/mc.duckdb")

# --- Constants ---
INVESTMENT_RATIO = 1
MANAGE_FEE_RATIO = 0.008        # annual fee
BUY_COST_RATIO = 0.006208
SELL_COST_RATIO = 0.006208
REBALANCE_FREQ = "quarterly"  # options: "daily", "weekly", "monthly", "quarterly"

# --- Market data loaders ---
def load_database_dic(codes: list[str]):
    price_dic = {}
    con = duckdb.connect(str(MC_DUCKDB))
    for code in codes:
        try:
            df = pl.from_arrow(
                con.execute(
                    f"SELECT date, op, cl, vol FROM mc WHERE code = '{code}'"
                ).arrow()
            )
            df = df.with_columns(pl.col("date").cast(pl.Date))
            price_dic[code] = {
                d: {"op": o, "cl": c, "vol": v}
                for d, o, c, v in zip(df["date"], df["op"], df["cl"], df["vol"])
            }
        except Exception as e:
            print(f"Error for {code}: {e}")
    con.close()
    return price_dic

# --- Load strategy CSV ---
def load_strategy_csv(path):
    strategy_name = os.path.splitext(os.path.basename(path))[0]
    df = pl.read_csv(
        path,
        dtypes={"code": pl.Utf8}
    ).with_columns([
        pl.lit(strategy_name).alias("strategy_name"),
        pl.col("code").cast(pl.Utf8),
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
    ])
    print("strategy_name:", strategy_name)
    return df, strategy_name

# --- Performance metrics ---
def compute_metrics(nav_series, dates):
    nav = np.array(nav_series)
    nav = nav[nav > 0]  # ensure positivity
    returns = np.diff(nav) / nav[:-1]
    total_return = nav[-1] / nav[0] - 1
    years = (len(dates) / 252)
    annualized_return = (1 + total_return) ** (1/years) - 1
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    cummax = np.maximum.accumulate(nav)
    drawdowns = nav / cummax - 1
    max_drawdown = np.min(drawdowns)
    return {
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown
    }
