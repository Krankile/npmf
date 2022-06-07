class MCLoss:
    mse = "mse_2"
    mape = "mape_2"


class VolLoss:
    std_diff = "std_diff"


class FundamentalLoss:
    pass


class MarketCap:
    name = "market_cap"
    loss = MCLoss


class Volatility:
    name = "volatility"
    loss = VolLoss


class Fundamentals:
    name = "fundamentals"
    loss = FundamentalLoss


class Problem:
    market_cap = MarketCap
    volatility = Volatility
    fundamentals = Fundamentals
