import math
from datetime import timedelta
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from torch.utils.data import Dataset

from .dtypes import fundamental_types


def get_stocks_in_timeframe(
    stock_df, stock_dates, scale=True, remove_na=True
) -> pd.DataFrame:
    out = pd.DataFrame(
        data=0, columns=stock_dates, index=stock_df.ticker.unique(), dtype=np.float64
    )
    stock_df = stock_df.pivot(index="ticker", columns="date", values="market_cap")
    out = out.add(stock_df)

    # Remove tickers where data missing for ticker in the whole period
    out = out.dropna(axis=0, how="all")

    if remove_na:
        out: pd.DataFrame = out.ffill(axis=1).replace(np.nan, 0)

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
        fundamental_df.loc[:, f"revenue_q={-q}":f"fcf_q={-q}"] = (
            fundamental_df.loc[:, f"revenue_q={-q}":f"fcf_q={-q}"]
            .div(last_market_cap_col, axis=0)
            .clip(upper=3, lower=-3)
        )
        fundamental_df.loc[
            :, f"total_assets_q={-q}":f"total_current_liabilities_q={-q}"
        ] = fundamental_df.loc[
            :, f"total_assets_q={-q}":f"total_current_liabilities_q={-q}"
        ].div(
            fundamental_df.loc[:, f"total_assets_q={-q}"], axis=0
        )
        fundamental_df.loc[
            :, f"long_term_debt_p_assets_q={-q}":f"short_term_debt_p_assets_q={-q}"
        ] = fundamental_df.loc[
            :, f"long_term_debt_p_assets_q={-q}":f"short_term_debt_p_assets_q={-q}"
        ].div(
            100
        )
        fundamental_df = fundamental_df.drop(columns=f"total_assets_q={-q}")

    fundamental_df = fundamental_df.replace(np.nan, 0)

    return fundamental_df


def get_last_q_fundamentals(fundamental_df, q):
    fundamental_df = fundamental_df[~fundamental_df.date.isna()].astype(
        fundamental_types
    )
    tickers = fundamental_df.ticker.unique()

    fundamental_df["rank"] = (
        fundamental_df.groupby("ticker")
        .date.rank(method="first", ascending=False)
        .astype(int)
    )
    fundamental_df = fundamental_df.set_index(["ticker", "rank"])
    fundamental_df = fundamental_df[fundamental_df.index.get_level_values(1) <= 4].loc[
        :, "revenue":
    ]

    multidx = pd.MultiIndex.from_product(
        [tickers, range(q, 0, -1)], names=["ticker", "rank"]
    )
    funds = pd.DataFrame(
        data=0,
        index=multidx,
        columns=fundamental_df.loc[:, "revenue":].columns,
        dtype=fundamental_df.dtypes.values,
    )

    result = funds.add(fundamental_df).sort_index(ascending=[True, False])
    return result


def get_fundamentals(fundamental_df, stock_tickers, current_time, n_reports):
    # Only keep fundamentals for where we have stock data
    legal_fundamental_df = fundamental_df[
        (fundamental_df.announce_date < current_time)
        & (fundamental_df.ticker.isin(stock_tickers))
        & ~fundamental_df.date.isna()
    ]

    # Important dimensions
    n_companies_with_fundamentals = len(legal_fundamental_df.ticker.unique())
    m_fundamentals = legal_fundamental_df.loc[:, "revenue":].shape[1]

    # Get last q fundamentals and return NA rows if they are still missing
    fundamental_df_all_quarters = get_last_q_fundamentals(
        legal_fundamental_df, n_reports
    )
    fundamentals = fundamental_df_all_quarters.to_numpy().reshape(
        (n_companies_with_fundamentals, n_reports * m_fundamentals)
    )

    return fundamentals, legal_fundamental_df


def get_meta_df(meta_df: pd.DataFrame, stocks_and_fundamentals: pd.DataFrame):
    legal_meta_df: pd.DataFrame = meta_df.set_index("ticker")

    # Join meta and stock-fundamentals
    legal_meta_df = legal_meta_df.loc[stocks_and_fundamentals.index, :]
    legal_meta_df.loc[:, "exchange_code":"state_province_hq"] = legal_meta_df.loc[
        :, "exchange_code":"state_province_hq"
    ].astype("category")
    legal_meta_df.loc[:, "economic_sector":"activity"] = legal_meta_df.loc[
        :, "economic_sector":"activity"
    ].astype("category")

    meta_cont = legal_meta_df["founding_year"].astype(np.float64)

    meta_cont = meta_cont.replace(to_replace=np.nan, value=meta_cont.mean(skipna=True))
    meta_cont = (meta_cont / 2000).to_frame()

    cat_cols = legal_meta_df.select_dtypes("category").columns
    meta_cat = legal_meta_df[cat_cols].apply(lambda col: col.cat.codes) + 1

    return meta_cont.astype(np.float32), meta_cat.astype(np.int64)


def normalize_macro(legal_macro_df, macro_df):
    df = legal_macro_df.copy()
    for column in [c for c in legal_macro_df.columns if ("_fx" not in c)]:
        df[column] = legal_macro_df[column] / (
            int(math.ceil(macro_df[column].max() / 100.0)) * 100
        )
    return df


def get_macro_df(
    macro_df: pd.DataFrame, historic_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    macro_df = macro_df.set_index("date")

    legal_macro_df = macro_df.loc[macro_df.index.isin(historic_dates), :]

    full_macro_df = pd.DataFrame(
        data=legal_macro_df, index=historic_dates, columns=legal_macro_df.columns
    ).ffill(axis=0)
    full_macro_df = normalize_macro(full_macro_df, macro_df).replace(np.nan, 0)
    return full_macro_df.astype(np.float32)


def get_forecast(
    stock_df: pd.DataFrame,
    stocks_and_fundamentals: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    last_market_cap_col: pd.Series,
):

    forecasts: pd.DataFrame = stock_df[stock_df.date.isin(forecast_dates)]

    forecasts_unnormalized = get_stocks_in_timeframe(
        forecasts,
        forecast_dates,
        scale=False,
        remove_na=False,
    )
    tickers = stocks_and_fundamentals.index.intersection(forecasts_unnormalized.index)
    forecasts_unnormalized = forecasts_unnormalized.loc[tickers, :]

    # TODO: Check if using the same MinMax-scaler as for training set is better
    forecasts_normalized = forecasts_unnormalized.div(
        last_market_cap_col.loc[tickers], axis=0
    )

    forecasts_normalized = forecasts_normalized.astype(np.float32)

    return forecasts_normalized


class TimeDeltaDataset(Dataset):
    def __init__(
        self,
        current_time: pd.Timestamp,
        training_window: int,
        forecast_window: int,
        n_reports: int,
        stock_df: pd.DataFrame,
        fundamental_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        macro_df: pd.DataFrame,
    ):
        # Get the relevant dates for training and forecasting
        historic_dates = get_historic_dates(current_time, training_window)
        forecast_dates = get_forecast_dates(current_time, forecast_window)

        # Get stock df
        legal_stock_df = stock_df.copy().loc[stock_df.date.isin(historic_dates), :]
        formatted_stocks = get_stocks_in_timeframe(
            legal_stock_df, historic_dates, scale=True, remove_na=True
        )

        # Get relative size information
        (
            relative_to_global_market_column,
            relative_to_current_market_column,
            last_market_cap_col,
        ) = get_global_local_column(legal_stock_df)

        # Get fundamentals df
        stock_tickers: np.array = legal_stock_df.ticker.unique()
        fundamentals, legal_fundamental_df = get_fundamentals(
            fundamental_df, stock_tickers, current_time, n_reports
        )
        fundamental_df = create_fundamental_df(
            fundamentals,
            legal_fundamental_df,
            n_reports,
            relative_to_current_market_column,
            relative_to_global_market_column,
            last_market_cap_col,
        )

        # Combine stocks and fundamentals
        # TODO: Review the strategy for dealing with nan values
        stocks_and_fundamentals = (
            formatted_stocks.join(fundamental_df).replace(np.nan, 0).astype(np.float32)
        )

        # Get forecasts
        self.forecast = get_forecast(
            stock_df, stocks_and_fundamentals, forecast_dates, last_market_cap_col
        )
        self.stocks_and_fundamentals = stocks_and_fundamentals.loc[
            self.forecast.index, :
        ]

        # Get meta df
        self.meta_cont, self.meta_cat = get_meta_df(
            meta_df, self.stocks_and_fundamentals
        )

        # Get macro df
        self.macro_df = get_macro_df(macro_df, historic_dates)

    def __len__(self):
        return self.stocks_and_fundamentals.shape[0]

    def __getitem__(self, idx):

        return (
            self.stocks_and_fundamentals.iloc[idx, :].to_numpy(),
            self.meta_cont.iloc[idx, :].to_numpy(),
            self.meta_cat.iloc[idx, :].to_numpy(),
            self.macro_df.T.to_numpy().ravel(),
            self.forecast.iloc[idx, :].to_numpy(),
        )
