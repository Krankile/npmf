from collections import namedtuple

import pandas as pd

from ...utils import Problem


def register_na_percentage(dictionary: dict, df_nick_name: str, df: pd.DataFrame):
    dictionary[df_nick_name] = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])


RelativeCols = namedtuple("RelativeCols", field_names="global_ current last")


def clamp_and_slice(dataset, *, conf):
    if conf.get("clamp") is not None:
        clamp = conf.clamp
        dataset.data = dataset.data.clip(-clamp, clamp)

    if conf.get("feature_subset") is not None:
        dataset.data = dataset.data[:, conf.feature_subset, :]

    if (
        conf.forecast_problem == Problem.fundamentals.name
        and conf.get("fundamental_targets") is not None
    ):
        dataset.target = dataset.target[:, conf.fundamental_targets]

    return dataset
