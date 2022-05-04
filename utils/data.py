import pandas as pd


# Defined globally where the test set starts in the complete dataset
test_start_str = "2019-01-01"
test_start_pd = pd.to_datetime(test_start_str)


def get_1_month_test_set(df):
    out = df[df.index >= test_start_pd]
    out["obs_number"] = (
        out.groupby("ticker").date.rank(method="first", ascending=True).astype(int)
    )
    return out[out.obs_number <= 20].drop(columns=["obs_number"])
