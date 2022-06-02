import numpy as np
import pandas as pd

from ..wandb import get_dataset
from . import test_start_pd


def get_all_data():
    all_stck = (
        get_dataset("stock-data:final", "master")
        .set_index(["ticker", "date"])
        .sort_index()
    )
    all_fund = (
        get_dataset("fundamental-data:final", "master")
        .astype(
            {
                "date": np.datetime64,
                "period_end_date": np.datetime64,
                "announce_date": np.datetime64,
            }
        )
        .set_index(["ticker", "announce_date"])
        .drop(columns=["period_end_date"])
        .sort_index()
    )

    all_macr = get_dataset("macro-data:final", "master").set_index("date").sort_index()

    print(
        f"Shape stock: {all_stck.shape}, shape fundamental: {all_fund.shape}, shape macro: {all_macr.shape}"
    )

    return all_stck, all_fund, all_macr


def get_training_data(all_data: list = None):

    if all_data is None:
        s, f, m = get_all_data()
    else:
        s, f, m = all_data

    stck = s[s.index.get_level_values(1) < test_start_pd]
    fund = f[f.index.get_level_values(1) < test_start_pd]
    macr = m[m.index < test_start_pd]

    return stck, fund, macr


def get_arimax_training_data(
    stck: pd.DataFrame,
    fund: pd.DataFrame,
    macr: pd.DataFrame,
    ticker: str,
    tw: int = 60,
):
    s = stck.loc[(ticker,)]
    f = fund.loc[(ticker,)]

    s = (
        s.join(f, how="left", rsuffix="_fund")
        .join(macr, how="left", rsuffix="_macr")
        .ffill()
    )
    s = s.iloc[-tw:, :]

    ys = s.market_cap
    exs = s.drop(columns=["market_cap", "close_price", "currency", "date"])
    exs = exs.loc[:, exs.nunique() > 1].dropna()

    return ys, exs
