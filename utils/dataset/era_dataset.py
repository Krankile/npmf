import math
from datetime import timedelta
from functools import partial
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, minmax_scale
from torch.utils.data import Dataset

from ...utils import Problem
from ...utils.dataset.utils import RelativeCols, register_na_percentage
from ..dtypes import fundamental_types


def get_stocks_in_timeframe(stock_df, stock_dates, remove_na=True) -> pd.DataFrame:
    out = pd.DataFrame(
        data=0, columns=stock_dates, index=stock_df.ticker.unique(), dtype=np.float32
    )
    stock_df = stock_df.pivot(index="ticker", columns="date", values="market_cap")
    out = out.add(stock_df)

    # Remove tickers where data missing for ticker in the whole period
    out = out.dropna(axis=0, how="all")

    if remove_na:
        out: pd.DataFrame = out.ffill(axis=1).replace(np.nan, 0)

    return out


def get_historic_dates(current_time, trading_days):
    back_in_time_buffer = timedelta(trading_days + trading_days * 5)

    return pd.date_range(
        start=current_time - back_in_time_buffer, end=current_time, freq="B"
    )[-trading_days:]


def get_target_dates(current_time: np.datetime64, forecast_w: int) -> pd.DatetimeIndex:
    forward_in_time_buffer = timedelta(forecast_w + forecast_w * 5)
    return pd.date_range(
        start=current_time + timedelta(1),
        end=current_time + forward_in_time_buffer,
        freq="B",
    )[:forecast_w]


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

    return RelativeCols(
        relative_to_global_market_column,
        relative_to_current_market_column,
        last_market_cap_col,
    )


def normalize_fundamentals(
    f: pd.DataFrame,
    relatives: RelativeCols,
):
    f = f.reindex(columns=(["global_relative", "peers_relative"] + f.columns.to_list()))
    f = f.reset_index().set_index("ticker")

    f["global_relative"] = relatives.global_
    f["peers_relative"] = relatives.current

    f.loc[:, "revenue":"fcf"] = (f.loc[:, "revenue":"fcf"] / relatives.last).clip(
        lower=-3, upper=3
    )

    total_assets = f.loc[:, "total_assets"]
    f = f.drop(columns="total_assets")

    for col in f.loc[:, "total_current_assets":"total_current_liabilities"].columns:
        f[col] /= total_assets

    f.loc[:, "long_term_debt_p_assets":"short_term_debt_p_assets"] = (
        f.loc[:, "long_term_debt_p_assets":"short_term_debt_p_assets"] / 100
    )
    return f


def get_3d_fundamentals(
    fundamental_df: pd.DataFrame,
    tickers: pd.Index,
    historic_dates,
    register_na,
    relatives: RelativeCols,
):
    current_time = historic_dates.values[-1]
    f = fundamental_df[
        (current_time + pd.Timedelta(weeks=104) <= fundamental_df.announce_date)
        & (fundamental_df.announce_date <= current_time)
    ]

    f = fundamental_df.set_index(["ticker", "announce_date"]).drop(columns=["date"])
    f = f.groupby(level=f.index.names).last()

    rest = tickers.difference(f.index.get_level_values(0).unique())
    missing = pd.DataFrame(
        np.nan,
        index=pd.MultiIndex.from_product(
            [rest, range(4)], names=["ticker", "announce_date"]
        ),
        columns=f.columns,
    )

    register_na(pd.concat([f, missing], axis=0))

    f = normalize_fundamentals(f, relatives)

    dates = pd.date_range(end=current_time, periods=240 * 2, freq="D")
    f = f.reset_index().set_index(["ticker", "announce_date"])
    f.index = f.index.rename("date", level=1)
    f = f.reindex(
        index=pd.MultiIndex.from_product([tickers, dates], names=["ticker", "date"])
    )
    f = f.loc[f.index.get_level_values(1).isin(historic_dates), :]
    f = f.groupby(level=0).ffill().replace(np.nan, 0)

    return f


def get_meta_df(meta_df: pd.DataFrame, tickers: pd.Index):
    legal_meta_df: pd.DataFrame = meta_df.set_index("ticker")

    # Join meta and stock-fundamentals
    legal_meta_df = legal_meta_df.loc[tickers, :]
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


def stock_target(
    stock_df: pd.DataFrame, tickers: pd.Index, target_dates
):
    targets: pd.DataFrame = stock_df[stock_df.date.isin(target_dates)]

    targets_unnormalized = get_stocks_in_timeframe(
        targets,
        target_dates,
        remove_na=False,
    )

    tickers = tickers.intersection(targets_unnormalized.index.unique()).sort_values()
    targets_unnormalized = targets_unnormalized.loc[tickers, :].astype(np.float32)

    return targets_unnormalized, tickers


def fundamental_target(
    fundamental_df: pd.DataFrame,
    tickers: pd.Index,
    target_dates,
):

    targets: pd.DataFrame = (
        fundamental_df[fundamental_df.date <= target_dates[-1]]
        .dropna(how="all")
        .groupby("ticker")
        .last()
    )
    last: pd.DataFrame = (
        fundamental_df[fundamental_df.date < target_dates[0]]
        .dropna(how="all")
        .groupby("ticker")
        .last()
    )

    constant = targets.replace(np.nan, 0).eq(last.replace(np.nan, 0)).all(axis=1)

    targets = targets.loc[~constant]

    tickers = tickers.intersection(targets.index.unique()).sort_values()
    targets = targets.drop(columns=["date", "announce_date"]).loc[tickers, :]

    return targets, tickers


def normalize_stock_target(
    target, tickers, scaler, last_market_cap_col, normalize_targets
):
    if normalize_targets == Problem.market_cap.normalize.mcap:
        return target.div(last_market_cap_col.loc[tickers], axis=0)
    elif normalize_targets == Problem.market_cap.normalize.minmax:
        return pd.DataFrame(
            data=scaler.transform(target.values.T).T,
            index=target.index,
            columns=target.columns,
        )

    raise ValueError


def normalize_target(
    target,
    forecast_problem,
    tickers,
    relatives,
    scaler=None,
    normalize_targets=None,
):
    if forecast_problem == Problem.fundamentals.name:
        return normalize_fundamentals(target, relatives)

    return normalize_stock_target(
        target, tickers, scaler, relatives.last, normalize_targets
    )


def get_target(
    stock_df: pd.DataFrame,
    fundamental_df: pd.DataFrame,
    tickers: pd.Index,
    target_dates: pd.DatetimeIndex,
    forecast_problem: str,
):

    if forecast_problem == Problem.fundamentals.name:
        return fundamental_target(fundamental_df, tickers, target_dates)

    return stock_target(stock_df, tickers, target_dates)


class EraDataset(Dataset):
    def __init__(
        self,
        *,
        current_time: pd.Timestamp,
        training_w: int,
        forecast_w: int,
        stock_df: pd.DataFrame,
        fundamental_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        forecast_problem: str,
        normalize_targets: str,
        **_,
    ):
        # Get the relevant dates for training and targeting
        historic_dates = get_historic_dates(current_time, training_w)
        target_dates = get_target_dates(current_time, forecast_w)

        # Initialize NA counter
        self.na_percentage = dict()

        # Get stock df
        legal_stock_df = (
            stock_df.astype(
                {
                    col: fundamental_types[col]
                    for col in stock_df.columns
                    if col in fundamental_types
                }
            )
            .copy()
            .loc[stock_df.date.isin(historic_dates), :]
        )
        register_na_percentage(self.na_percentage, "stock", legal_stock_df)

        formatted_stocks = get_stocks_in_timeframe(
            legal_stock_df,
            historic_dates,
            remove_na=True,
        )

        # Get relative size information
        relatives = get_global_local_column(legal_stock_df)

        tickers: pd.Index = formatted_stocks.index.unique()

        # Get targets
        target, tickers = get_target(
            stock_df,
            fundamental_df,
            tickers,
            target_dates,
            forecast_problem,
        )
        register_na_percentage(self.na_percentage, "target", self.target)

        # Make sure that only tickers with data both in training and forecasting is included
        formatted_stocks = formatted_stocks.loc[tickers, :]

        scaler = MinMaxScaler()
        formatted_stocks = pd.DataFrame(
            data=scaler.fit_transform(formatted_stocks.values.T).T,
            index=formatted_stocks.index,
            columns=formatted_stocks.columns,
        )

        target = normalize_target(
            target, forecast_problem, tickers, relatives, scaler, normalize_targets
        )

        legal_fundamentals = get_3d_fundamentals(
            fundamental_df,
            tickers,
            historic_dates,
            partial(register_na_percentage, self.na_percentage, "fundamental"),
            relatives,
        )

        # TODO: Review the strategy for dealing with nan values

        # Get meta df
        self.meta_cont, self.meta_cat = get_meta_df(meta_df, tickers)

        # Get macro df
        register_na_percentage(self.na_percentage, "macro", macro_df)
        macro_df = get_macro_df(macro_df, historic_dates)

        self.data_fields = (
            ["market_cap"]
            + legal_fundamentals.columns.to_list()
            + macro_df.columns.to_list()
        )

        self.target = target
        self.target_fields = target.columns.to_list()

        self.target = self.target.to_numpy().astype(np.float32)

        self.tickers = formatted_stocks.index.to_list()

        formatted_stocks = formatted_stocks.to_numpy().reshape((-1, training_w, 1))
        legal_fundamentals = legal_fundamentals.to_numpy().reshape((-1, training_w, 18))

        macro_df = (
            macro_df.to_numpy()
            .reshape((1, training_w, 18))
            .repeat(len(tickers), axis=0)
        )

        self.data: np.ndarray = (
            np.concatenate((formatted_stocks, legal_fundamentals, macro_df), axis=2)
            .transpose(0, 2, 1)
            .astype(np.float32)
        )

    def __len__(self):
        return self.data.shape[0]

    @property
    def n_features(self):
        return self.data.shape[1]

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        return (
            self.data[idx, :, :],
            self.meta_cont.iloc[idx, :].to_numpy(),
            self.meta_cat.iloc[idx, :].to_numpy(),
            self.target[idx, :],
        )
