import numpy as np


def mape(actual, forecast):
    return ((np.abs(actual - forecast)) / np.abs(actual)).mean()
