class MCLoss:
    mse = "mse_2"
    mape = "mape_2"


class VolLoss:
    std_diff_mae = "std_diff_mae"
    std_diff_mse = "std_diff_mse"


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


class FundTrainingW:
    h240 = 240
    h480 = 480


class MCTrainingW:
    h240 = 240
    h480 = 480


class VolTrainingW:
    h240 = 240
    h480 = 480


class MCNormalize:
    mcap = "mcap"
    minmax = "minmax"


class MarketCap:
    name = "market_cap"
    loss = MCLoss
    training_w = MCTrainingW
    forecast_w = MCForecastW
    normalize = MCNormalize


class Volatility:
    name = "volatility"
    loss = VolLoss
    training_w = VolTrainingW
    forecast_w = VolForecastW


class Fundamentals:
    name = "fundamentals"
    loss = FundLoss
    training_w = FundTrainingW
    forecast_w = FundForecastW


class Problem:
    market_cap = MarketCap
    volatility = Volatility
    fundamentals = Fundamentals
