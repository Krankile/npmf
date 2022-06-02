import pandas as pd


# Defined globally where the test set starts in the complete dataset
test_start_str = "2019-01-01"
test_start_pd = pd.to_datetime(test_start_str)


def get_1_month_test_set(df, limit=20, date_field="date"):
    out = df[df.index >= test_start_pd].copy()
    out.loc[:, "obs_number"] = (
        out.groupby("ticker")[date_field]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    return out[out.obs_number <= limit].drop(columns=["obs_number"])


def get_train_set(df, limit=None, date_field="date"):
    out = df[df.index < test_start_pd].copy()

    if limit is not None:
        out.loc[:, "obs_number"] = (
            out.groupby("ticker")[date_field]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        out = out[out.obs_number <= limit].drop(columns=["obs_number"])

    return out
