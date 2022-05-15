
col_map_macro = {
    "BRT-": "brt",
    "CLc1": "clc",
    "WTCLc1": "wtc",
    "LNG-AS": "lng",
    ".VIX": "vix",
    "EUR=": "eur_fx",
    "GBP=": "gbp_fx",
    "CNY=": "cny_fx",
}

col_map_meta =  {"Instrument": "ticker",
    "Date": "date",
    "Exchange Market Identifier Code": "exchange_code",
    "Region of Headquarters": "region_hq",
    "Country of Headquarters": "country_hq",
    "State or Province of Headquarters": "state_province_hq",
    "Organization Founded Year": "founding_year",
    "TRBC Economic Sector Name": "economic_sector",
    "TRBC Business Sector Name": "business_sector",
    "TRBC Industry Group Name": "industry_group",
    "TRBC Industry Name": "industry",
    "TRBC Activity Name": "activity",}

col_map_fundamentals =  {"Instrument": "ticker",
    "Date": "date",
    "Period End Date": "period_end_date",
    "Balance Sheet Orig Announce Date": "announce_date",
    "Total Revenue": "revenue",
    "Gross Profit": "gross_profit",
    "EBITDA": "ebitda",
    "EBIT": "ebit",
    "Net Income after Tax": "net_income",
    "Free Cash Flow": "fcf",
    "Total Assets": "total_assets",
    "Total Current Assets": "total_current_assets",
    "Total Liabilities": "total_liabilites",
    "Total Current Liabilities": "total_current_liabilities",
    "Long Term Debt Percentage of Total Assets": "long_term_debt_p_assets",
    "Short Term Debt Percentage of Total Assets": "short_term_debt_p_assets",
    "Gross Profitp": "gross_profit_p",
    "EBITDAp": "ebitda_p",
    "EBITp": "ebit_p",
    "Net Income after Taxp": "net_income_p",}

col_map_stock = {
    "Instrument": "ticker",
    "Date": "date",
    "Company Market Cap": "market_cap",
    "Price Close": "close_price",
    "Currency": "currency",
}

column_mapping = {**col_map_macro, **col_map_meta, **col_map_fundamentals, **col_map_stock}
