import numpy as np


fundamental_types = {
    "ticker": str,
    "date": np.datetime64,
    "announce_date": np.datetime64,
    "revenue": np.float32,
    "gross_profit": np.float32,
    "ebitda": np.float32,
    "ebit": np.float32,
    "net_income": np.float32,
    "fcf": np.float32,
    "total_assets": np.float32,
    "total_current_assets": np.float32,
    "total_liabilites": np.float32,
    "total_current_liabilities": np.float32,
    "long_term_debt_p_assets": np.float32,
    "short_term_debt_p_assets": np.float32,
    "gross_profit_p_revenue": np.float32,
    "ebitda_p_revenue": np.float32,
    "ebit_p_revenue": np.float32,
    "net_income_p_revenue": np.float32,
    "fcf_p_revenue": np.float32,
}
