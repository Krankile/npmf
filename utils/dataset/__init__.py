from collections import namedtuple

import pandas as pd

from .era_controller import EraController
from .era_dataset import EraDataset
from .timedelta_dataset import TimeDeltaDataset


def register_na_percentage(dictionary: dict, df_nick_name: str, df: pd.DataFrame):
    dictionary[df_nick_name] = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])


RelativeCols = namedtuple("RelativeCols", field_names="global_ current last")


__all__ = [
    EraDataset,
    TimeDeltaDataset,
    EraController,
    register_na_percentage,
    RelativeCols,
]
