import pandas as pd

from ...utils.dataset import get_stocks_in_timeframe


def test_get_stocks_in_timeframe():
    stock_df = pd.read_excel("stock_df.xlsx")

    get_stocks_in_timeframe()
