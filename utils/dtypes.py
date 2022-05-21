import numpy as np


fundamental_types = {
    "ticker": str,
    "date": np.datetime64,
    "period_end_date": np.datetime64,
    "announce_date": np.datetime64,
    "revenue": np.float64,
    "gross_profit": np.float64,
    "ebitda": np.float64,
    "ebit": np.float64,
    "net_income": np.float64,
    "fcf": np.float64,
    "total_assets": np.float64,
    "total_current_assets": np.float64,
    "total_liabilites": np.float64,
    "total_current_liabilities": np.float64,
    "long_term_debt_p_assets": np.float64,
    "short_term_debt_p_assets": np.float64,
    "gross_profit_p": np.float64,
    "ebitda_p": np.float64,
    "ebit_p": np.float64,
    "net_income_p": np.float64,
}
