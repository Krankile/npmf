class MCLoss:
    mse = "mse_2"
    mape = "mape_2"


class VolLoss:
    std_diff = "std_diff"


class FundLoss:
    mape = "mape_2"
    smape = "smape"
    ce_bankruptcy = "ce_bankruptcy"


class FundForecastW:
    h60 = 60
    h240 = 240


class MCForecastW:
    h20 = 20
    h240 = 240


class VolForecastW:
    h20 = 20


class MCNormalize:
    mcap = "mcap"
    minmax = "minmax"


class MarketCap:
    name = "market_cap"
    loss = MCLoss
    forecast_w = MCForecastW
    normalize = MCNormalize


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
