from datetime import timedelta
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale

from ..utils.dtypes import fundamental_types


def get_stocks_in_timeframe(stock_df, stock_dates, scale=True, remove_na=True):
    out = pd.DataFrame(
        data=0, columns=stock_dates, index=stock_df.ticker.unique(), dtype=np.float64
    )
    stock_df = stock_df.pivot(index="ticker", columns="date", values="market_cap")
    out = out.add(stock_df)

    if remove_na:
        out = out.ffill(axis=1)
        out[out.isna()] = 0

    # Perform MinMaxScaling on the full dataset
    if scale:
        out = pd.DataFrame(
            data=minmax_scale(out.values, axis=1),
            index=out.index,
            columns=out.columns,
        )
    return out


def get_historic_dates(current_time, trading_days):
    back_in_time_buffer = timedelta(trading_days + trading_days * 5)

    return pd.date_range(
        start=current_time - back_in_time_buffer, end=current_time, freq="B"
    )[-trading_days:]


def get_forecast_dates(
    current_time: np.datetime64, forecast_window: int
) -> pd.DatetimeIndex:
    forward_in_time_buffer = timedelta(forecast_window + forecast_window * 5)
    return pd.date_range(
        start=current_time + timedelta(1),
        end=current_time + forward_in_time_buffer,
        freq="B",
    )[:forecast_window]


def _get_last_market_cap(stock_df: pd.DataFrame) -> pd.Series:
    return (
        stock_df.dropna(subset=["market_cap"])
        .drop_duplicates(subset=["ticker"], keep="last")
        .set_index("ticker")
        .market_cap.squeeze()
        .astype(np.float64)
    )


def _minmax_scale_series(series: pd.Series) -> pd.Series:
    return pd.Series(
        minmax_scale(series.to_numpy().reshape((-1, 1))).squeeze(),
        index=series.index,
    )


def get_global_local_column(
    stock_df: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    apple_market_cap = 2.687 * (10**12)  # ish as of may 2022 (USD)

    last_market_cap_col = _get_last_market_cap(stock_df)

    relative_to_global_market_column: pd.Series = last_market_cap_col / apple_market_cap
    relative_to_current_market_column = _minmax_scale_series(last_market_cap_col)

    return (
        relative_to_global_market_column,
        relative_to_current_market_column,
        last_market_cap_col,
    )


def create_fundamental_df(
    fundamentals,
    legal_fundamental_df,
    n_reports,
    relative_to_current_market_column,
    relative_to_global_market_column,
    last_market_cap_col,
):
    fund_columns = []
    for i in range(n_reports):
        fund_columns.extend(
            legal_fundamental_df.loc[0, "revenue":]
            .index.to_series()
            .map(lambda title: f"{title}_q=-{n_reports-i}")
        )
    columns = ["global_relative"] + ["peers_relative"] + fund_columns
    fundamental_df = pd.DataFrame(
        index=legal_fundamental_df.ticker.unique(), columns=columns
    )

    fundamental_df["peers_relative"] = relative_to_current_market_column.loc[
        fundamental_df.index
    ]
    fundamental_df["global_relative"] = relative_to_global_market_column.loc[
        fundamental_df.index
    ]

    fundamental_df.loc[:, f"revenue_q={-n_reports}":"net_income_p_q=-1"] = fundamentals
    for q in range(n_reports, 0, -1):
        fundamental_df.loc[:, f"revenue_q={-q}":f"fcf_q={-q}"] = fundamental_df.loc[
            :, f"revenue_q={-q}":f"fcf_q={-q}"
        ].div(last_market_cap_col, axis=0)
        fundamental_df.loc[
            :, f"total_assets_q={-q}":f"total_current_liabilities_q={-q}"
        ] = fundamental_df.loc[
            :, f"total_assets_q={-q}":f"total_current_liabilities_q={-q}"
        ].div(
            fundamental_df.loc[:, f"total_assets_q={-q}"], axis=0
        )
        fundamental_df = fundamental_df.drop(columns=f"total_assets_q={-q}")

    return fundamental_df


def get_last_q_fundamentals(fundamental_df, q):
    fundamental_df = fundamental_df[~fundamental_df.date.isna()].astype(fundamental_types)
    tickers = fundamental_df.ticker.unique()
    
    fundamental_df["rank"] = fundamental_df.groupby("ticker").date.rank(method="first", ascending=False).astype(int)
    fundamental_df = fundamental_df.set_index(["ticker", "rank"])
    fundamental_df = fundamental_df[fundamental_df.index.get_level_values(1) <= 4].loc[:, "revenue":]
    
    multidx = pd.MultiIndex.from_product([tickers, range(q, 0, -1)], names=["ticker", "rank"])
    funds = pd.DataFrame(data=0, index=multidx, columns=fundamental_df.loc[:, "revenue":].columns, dtype=fundamental_df.dtypes.values)
    
    result = funds.add(fundamental_df).sort_index(ascending=[True, False])
    return result 