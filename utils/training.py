from typing import List

import numpy as np
from tqdm import tqdm


class EarlyStop:
    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = patience
        self.min_delta = min_delta

        self.reset()

    def __call__(self, epoch_loss: List[float]) -> bool:
        loss = np.mean(epoch_loss)

        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.triggers = 0
        else:
            self.triggers += 1

        return not (self.triggers < self.patience)

    def reset(self):
        self.best_loss = float("inf")
        self.triggers = 0


def to_device(loader, device):
    for batch in loader:
        yield map(lambda data: data.to(device), batch)


def mape_loss(target, y_pred):
    mask = ~target.isnan()
    denom = mask.sum(dim=1)
    target[target != target] = 0
    l = ((((y_pred - target).abs() / (target.abs() + 1e-8) * mask)).sum(dim=1) / denom).mean()
    return l


class TqdmPostFix(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._postfix = dict()

    def update_postfix(self, d: dict):
        self._postfix.update(d)
        self.set_postfix(self._postfix)
