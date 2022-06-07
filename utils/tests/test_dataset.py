from pathlib import Path
import pandas as pd

from utils.dataset.timedelta_dataset import (
    get_global_local_column,
    get_stocks_in_timeframe,
    get_historic_dates,
    get_forecast_dates,
)

from .utils import unpickle_df

base_path = Path("utils", "tests", "data")


def test_get_stocks_in_timeframe_scaled():
    stock_df = unpickle_df(base_path / "stock_df.pickle")
    actual = unpickle_df(base_path / "out.pickle")

    current_time = pd.to_datetime("2010-03-01")
    trading_days = 240

    date_range = get_historic_dates(current_time, trading_days)

    out = get_stocks_in_timeframe(
        stock_df=stock_df, stock_dates=date_range, scale=True, remove_na=True
    )

    assert actual.equals(out)


def test_global_local_column():
    legal_stock_df = unpickle_df(base_path / "legal_stock_df.pickle")
    actual_global = unpickle_df(base_path / "relative_to_global_market_column.pickle")
    actual_local = unpickle_df(base_path / "relative_to_current_market_column.pickle")
    actual_last = unpickle_df(base_path / "last_market_cap_col.pickle")

    global_mc, local_mc, last_mc = get_global_local_column(stock_df=legal_stock_df)

    assert actual_global.equals(global_mc)
    assert actual_local.equals(local_mc)
    assert actual_last.equals(last_mc)
