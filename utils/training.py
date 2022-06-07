import math
from typing import List

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from ..utils import Problem


class TqdmPostFix(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._postfix = dict()

    def update_postfix(self, d: dict):
        self._postfix.update(d)
        self.set_postfix(self._postfix)


class EarlyStop:
    def __init__(
        self, patience: int, min_delta: float, model: nn.Module = None, pbar=None
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta

        self.model = model
        self.pbar = pbar

        self.reset()

    def __call__(self, epoch_loss: List[float]) -> bool:
        loss = np.mean(epoch_loss)

        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.triggers = 0

            if self.model is not None:
                self.best_wts = self.model.state_dict()
        else:
            self.triggers += 1

        if self.pbar is not None:
            self.pbar.update_postfix(
                {
                    "triggers": f"{self.triggers}/{self.patience}",
                    "best_loss": self.best_loss,
                }
            )

        stop_era = self.triggers >= self.patience

        if stop_era and self.model is not None:
            self.model.load_state_dict(self.best_wts)

        return stop_era

    def reset(self):
        self.best_loss = float("inf")
        self.triggers = 0
        self.best_wts = None
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


def mse_loss_2(target: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    mask = (~target.isnan()) & (target.abs() >= 1e-2)
    denom = mask.sum(dim=1)
    target[target != target] = 0
    l = ((((y_pred - target) ** 2 * mask)).sum(dim=1) / denom).mean()
    return l


def std_loss_diff_mse(target: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # y_t/y_k-y_{t-1}/y_k => (y_t-y_{t-1})/y_k * y_k/y_{t-1} = (y_t-y_{t-1})/y_{t-1}
    target_ = target.diff() * (target[:, :-1] ** (-1))

    mask = (~target_.isnan()) & (target_.abs() <= 10)
    mask2 = mask.sum(dim=1, keepdim=True) >= 2

    target_[~mask] = 0

    denom = mask.sum(dim=1, keepdim=True)
    denom2 = mask2.sum(dim=0).item()

    l = (
        (
            torch.nan_to_num(
                (
                    torch.sum(
                        (
                            (target_ - torch.sum(target_, dim=1, keepdim=True) / denom)
                            * mask
                        )
                        ** 2,
                        dim=1,
                        keepdim=True,
                    )
                    / denom
                )
                ** (1 / 2)
                - y_pred
            )
            * mask2
        )
        ** 2
    ).sum() / denom2

    return l


def std_loss_diff_abs(target: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # y_t/y_k-y_{t-1}/y_k => (y_t-y_{t-1})/y_k * y_k/y_{t-1} = (y_t-y_{t-1})/y_{t-1}
    target_ = target.diff() * (target[:, :-1] ** (-1))

    mask = (~target_.isnan()) & (target_.abs() <= 10)
    mask2 = mask.sum(dim=1, keepdim=True) >= 2

    target_[~mask] = 0

    denom = mask.sum(dim=1, keepdim=True)
    denom2 = mask2.sum(dim=0).item()

    l = (
        (
            torch.nan_to_num(
                (
                    torch.sum(
                        (
                            (target_ - torch.sum(target_, dim=1, keepdim=True) / denom)
                            * mask
                        )
                        ** 2,
                        dim=1,
                        keepdim=True,
                    )
                    / denom
                )
                ** (1 / 2)
                - y_pred
            )
            * mask2
        ).abs()
    ).sum() / denom2

    return l


loss_fns = dict(
    mape=mape_loss,
    mape_2=mape_loss_2,
    mse_2=mse_loss_2,
    std_diff=std_loss_diff_abs,
    std_diff_mse=std_loss_diff_mse,
)


activations = dict(
    relu=nn.ReLU,
    elu=nn.ELU,
    prelu=nn.PReLU,
    leaky=nn.LeakyReLU,
)

n_layers = lambda l, k, b: math.ceil(math.log((l - 1) * (b - 1) / ((k - 1) * 2) + 1, b))


def get_naive_pred(data, target, device, conf):
    if conf.forecast_problem == Problem.market_cap.name:
        return torch.ones(target.shape, device=device)

    if conf.forecast_problem == Problem.volatility.name:
        return data[:, 0, :-20].std(dim=1, keepdim=True)

    if conf.forecast_problem == Problem.fundamentals.name:
        return data[:, :, -1]

    raise ValueError("Invalid problem passed")
