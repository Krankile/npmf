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
    mask = (~target.isnan()) & target.isfinite()
    denom = mask.sum(dim=1)
    target[target != target] = 0
    l = (
        (((y_pred - target).abs() / (target.abs() + 1e-6) * mask)).sum(dim=1) / denom
    ).mean()
    return l.clamp(min=0, max=10_000)


def smape_loss(target: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    mask = (~target.isnan()) & target.isfinite()
    denom = mask.sum(dim=1)
    target[target != target] = 0
    l = (
        2
        * (((y_pred - target).abs() / (target.abs() + y_pred.abs() + 1e-6) * mask)).sum(
            dim=1
        )
        / denom
    ).mean()
    return l


def mse_loss_2(target: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:

    mask = ~target.isnan() & target.isfinite()
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


def std_loss_diff_mae(target: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
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


def three_balance_to_bankrupt(three_targets):
    # feature order: total_current_assets/total_assets  |   total_liabilites/total_assets  |    total_current_liabilities/total_assets
    assets_to_liab = 1 / three_targets[:, 1, :]
    asset_prob = assets_to_liab < 1

    curr_assets_to_liab = three_targets[:, 0, :] / three_targets[:, 2, :]
    curr_asset_prob = curr_assets_to_liab < 1

    both_problems = curr_asset_prob * asset_prob

    target = torch.sum(both_problems, dim=1, keepdim=True) > 1
    return target


def cross_entropy_bankruptcy(
    three_targets: torch.Tensor, y_pred: torch.Tensor
) -> torch.Tensor:
    target = three_balance_to_bankrupt(three_targets)

    weights = 1 / (
        target.sum().div(len(target)) * target
        + (~target).sum().div(len(target)) * (~target)
    )  # sheeeeeshhhh

    weights /= weights.sum()

    return nn.functional.binary_cross_entropy(
        torch.sigmoid(y_pred), target.to(torch.float32), weight=weights
    )

def cross_entropy_bankruptcy_unweighted(
    three_targets: torch.Tensor, y_pred: torch.Tensor
) -> torch.Tensor:
    target = three_balance_to_bankrupt(three_targets)

    return nn.functional.binary_cross_entropy(
        torch.sigmoid(y_pred), target.to(torch.float32)
    )


loss_fns = dict(
    mape=mape_loss,
    mape_2=mape_loss_2,
    mse_2=mse_loss_2,
    smape=smape_loss,
    std_diff_mae=std_loss_diff_mae,
    std_diff_mse=std_loss_diff_mse,
    ce_bankruptcy=cross_entropy_bankruptcy,
    ce_bankruptcy_unweighted=cross_entropy_bankruptcy_unweighted,
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
        if conf.get("normalize_targets") == Problem.market_cap.normalize.minmax:
            return data[:, 0, -1:]

        if (
            conf.get("normalize_targets") is None
            or conf.get("normalize_targets") == Problem.market_cap.normalize.mcap
        ):
            return torch.ones(target.shape, device=device)

    if conf.forecast_problem == Problem.volatility.name:
        return data[:, 0, -conf.forecast_w :].std(dim=1, keepdim=True)

    if conf.forecast_problem == Problem.fundamentals.name:
        if conf.train_loss == Problem.fundamentals.loss.ce_bankruptcy:
            slc = slice(9, 12) if conf.feature_subset is None else slice(0, 3)
            return three_balance_to_bankrupt(data[:, slc, -conf.forecast_w:]).to(torch.float32)

        subset = conf.get("fundamental_targets") or slice(1, 18 + 1)
        return data[:, subset, -1]

    raise ValueError("Invalid problem passed")
