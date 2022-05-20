import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale


def get_stocks_in_timeframe(stock_df, stock_dates, scale=True):
    out = pd.DataFrame(
        data=0, columns=stock_dates, index=stock_df.ticker.unique(), dtype=np.float64
    )
    stock_df = stock_df.pivot(index="ticker", columns="date", values="market_cap")
    out = out.add(stock_df).ffill(axis=1)

    out[out.isna()] = 0

    # Perform MinMaxScaling on the full dataset
    if scale:
        out = pd.DataFrame(
            data=minmax_scale(out.values, axis=1),
            index=out.index,
            columns=out.columns,
        )
    return out
