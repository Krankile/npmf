from pathlib import Path
import pandas as pd

from utils.dataset import (
    get_stocks_in_timeframe,
    get_historic_dates,
    get_forecast_dates,
)
from utils.tests import unpickle_df


def test_get_stocks_in_timeframe_scaled():
    base_path = Path("tests", "dataset")
    stock_df = unpickle_df(base_path / "stock_df.pickle")
    actual = unpickle_df(base_path / "out.pickle")

    current_time = pd.to_datetime("2010-03-01")
    trading_days = 240

    date_range = get_historic_dates(current_time, trading_days)

    out = get_stocks_in_timeframe(
        stock_df=stock_df, stock_dates=date_range, scale=True, remove_na=True
    )

    assert actual.equals(out)


def test_get_forecasts_in_timeframe():
    base_path = Path("tests", "dataset")
    forecasts = unpickle_df(base_path / "forecasts.pickle")
    actual = unpickle_df(base_path / "forecasts_unnormalized.pickle")

    current_time = pd.to_datetime("2010-03-01")
    forecast_window = 20

    date_range = get_forecast_dates(current_time, forecast_window)

    out = get_stocks_in_timeframe(
        stock_df=forecasts, stock_dates=date_range, scale=False, remove_na=False
    )

    assert actual.equals(out)
