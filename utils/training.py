import math
from typing import List

import numpy as np
import torch
from tqdm import tqdm


class TqdmPostFix(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._postfix = dict()

    def update_postfix(self, d: dict):
        self._postfix.update(d)
        self.set_postfix(self._postfix)


class EarlyStop:
    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = patience
        self.min_delta = min_delta

        self.reset()

    def __call__(self, epoch_loss: List[float], pbar: TqdmPostFix = None) -> bool:
        loss = np.mean(epoch_loss)

        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.triggers = 0
        else:
            self.triggers += 1

        if pbar is not None:
            pbar.update_postfix(
                {
                    "triggers": f"{self.triggers}/{self.patience}",
                    "best_loss": self.best_loss,
                }
            )

        return not (self.triggers < self.patience)

    def reset(self):
        self.best_loss = float("inf")
        self.triggers = 0
        return self


def to_device(loader, device):
    for batch in loader:
        yield map(lambda data: data.to(device), batch)


def mape_loss(target: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    mask = ~target.isnan()
    denom = mask.sum(dim=1)
    target[target != target] = 0
    l = (
        (((y_pred - target).abs() / (target.abs() + 1e-8) * mask)).sum(dim=1) / denom
    ).mean()
    return l

def mape_loss_2(target: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    mask = (~target.isnan()) & (target.abs() >= 1e-2)
    denom = mask.sum(dim=1)
    target[target != target] = 0
    l = (
        (((y_pred - target).abs() / (target.abs() + 1e-8) * mask)).sum(dim=1) / denom
    ).mean()
    return l

loss_fns = dict(
    mape=mape_loss,
    mape_2=mape_loss_2,
)


n_layers = lambda l, k, b: math.ceil(math.log((l - 1) * (b - 1) / ((k - 1) * 2) + 1, b))
