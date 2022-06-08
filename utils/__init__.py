class MCLoss:
    mse = "mse_2"
    mape = "mape_2"


class VolLoss:
    std_diff = "std_diff"


class FundLoss:
    mape = "mape_2"
    smape = "smape"


class FundForecastW:
    h60 = 60


class MCForecastW:
    h20 = 20
    h240 = 240


class VolForecastW:
    h20 = 20


class MarketCap:
    name = "market_cap"
    loss = MCLoss
    forecast_w = MCForecastW


class Volatility:
    name = "volatility"
    loss = VolLoss
    forecast_w = VolForecastW


class Fundamentals:
    name = "fundamentals"
    loss = FundLoss
    forecast_w = FundForecastW


class Problem:
    market_cap = MarketCap
    volatility = Volatility
    fundamentals = Fundamentals
