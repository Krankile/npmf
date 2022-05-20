from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale


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


def get_forecast_dates(current_time, forecast_window):
    forward_in_time_buffer = timedelta(forecast_window + forecast_window * 5)
    return pd.date_range(
        start=current_time + timedelta(1),
        end=current_time + forward_in_time_buffer,
        freq="B",
    )[:forecast_window]


def get_global_local_column(stock_df):
    last_market_cap_col = pd.Series()
    for ticker, df in tqdm(stock_df.groupby(by="ticker"), desc="Get global local column"):
        last_market_cap_col[ticker] = df.market_cap.dropna().iloc[-1]

    min_max_scaler = MinMaxScaler()
    
    #Add column to learn relative values
    apple_market_cap = 2.687*(10**12) #ish as of may 2022 (USD)
    
    relative_to_global_market_column: pd.Series = last_market_cap_col / apple_market_cap
    
    relative_to_current_market_column = min_max_scaler.fit_transform(last_market_cap_col.to_numpy().reshape((-1,1)))
    relative_to_current_market_column = pd.Series(relative_to_current_market_column[:,0], index=last_market_cap_col.index) 

    return relative_to_global_market_column, relative_to_current_market_column, last_market_cap_col