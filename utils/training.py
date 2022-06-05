import math
from typing import List

import numpy as np
import torch
from torch import nn
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


def mse_loss_2(target: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    mask = (~target.isnan()) & (target.abs() >= 1e-2)
    denom = mask.sum(dim=1)
    target[target != target] = 0
    l = ((((y_pred - target) ** 2 * mask)).sum(dim=1) / denom).mean()
    return l


def volatility_loss(target: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    mask = (~target.isnan()) & (target.abs() >= 1e-2)
    target[target != target] = 0
    denom = mask.sum(dim=1, keepdim=True)
    l = (
        (
            torch.sum(
                (target - torch.sum(target, dim=1, keepdim=True) / denom) ** 2,
                dim=1,
                keepdim=True,
            )
            * mask
            / denom
            - y_pred
        )
        ** 2
    ).mean()
    return l


def volatility_loss_abs(target: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    mask = (~target.isnan()) & (target.abs() >= 1e-2)
    target[target != target] = 0
    denom = mask.sum(dim=1, keepdim=True)
    l = (
        (
            torch.sum(
                (target - torch.sum(target, dim=1, keepdim=True) / denom) ** 2,
                dim=1,
                keepdim=True,
            )
            * mask
            / denom
            - y_pred
        ).abs()
    ).mean()
    return l

def volatility_loss_diff(target: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    #y_t/y_k-y_{t-1}/y_k => (y_t-y_{t-1})/y_k * y_k/y_{t-1} = (y_t-y_{t-1})/y_{t-1} 
    target_ = torch.nan_to_num(target.diff()*(target[:,:-1]**(-1)), posinf=np.nan, neginf=np.nan)
    
    mask = ((~target_.isnan()) & (target_.abs() <= 10) & (target_.abs() >= -10))
    target_[target_ != target_] = 0 
    denom = mask.sum(dim=1, keepdim=True)
    l = ((torch.sum((target_-torch.sum(target_, dim=1, keepdim=True)/denom)**2, dim=1, keepdim=True)*mask/denom - y_pred).abs()).mean()
    return l

def volatility_loss_diff_mse(target: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    #y_t/y_k-y_{t-1}/y_k => (y_t-y_{t-1})/y_k * y_k/y_{t-1} = (y_t-y_{t-1})/y_{t-1} 
    target_ = torch.nan_to_num(target.diff()*(target[:,:-1]**(-1)), posinf=np.nan, neginf=np.nan)
    
    mask = ((~target_.isnan()) & (target_.abs() <= 10) & (target_.abs() >= -10))
    target_[target_ != target_] = 0 
    denom = mask.sum(dim=1, keepdim=True)
    l = ((torch.sum((target_-torch.sum(target_, dim=1, keepdim=True)/denom)**2, dim=1, keepdim=True)*mask/denom - y_pred).abs()).mean()
    return l

def std_loss_diff_mse(target: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    #y_t/y_k-y_{t-1}/y_k => (y_t-y_{t-1})/y_k * y_k/y_{t-1} = (y_t-y_{t-1})/y_{t-1} 
    target_ = torch.nan_to_num(target.diff()*(target[:,:-1]**(-1)), posinf=np.nan, neginf=np.nan)
    
    mask = ((~target_.isnan()) & (target_.abs() <= 10) & (target_.abs() >= -10))
    target_[target_ != target_] = 0 
    denom = mask.sum(dim=1, keepdim=True)
    l = (((torch.sum((target_-torch.sum(target_, dim=1, keepdim=True)/denom)**2, dim=1, keepdim=True)*mask/denom)**(1/2) - y_pred)**2).mean()
    return l

def std_loss_diff_abs(target: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    #y_t/y_k-y_{t-1}/y_k => (y_t-y_{t-1})/y_k * y_k/y_{t-1} = (y_t-y_{t-1})/y_{t-1} 
    target_ = torch.nan_to_num(target.diff()*(target[:,:-1]**(-1)), posinf=np.nan, neginf=np.nan)
    
    mask = ((~target_.isnan()) & (target_.abs() <= 10) & (target_.abs() >= -10))
    target_[target_ != target_] = 0 
    denom = mask.sum(dim=1, keepdim=True)
    l = (((torch.sum((target_-torch.sum(target_, dim=1, keepdim=True)/denom)**2, dim=1, keepdim=True)*mask/denom)**(1/2) - y_pred).abs()).mean()
    return l    


loss_fns = dict(
    mape=mape_loss,
    mape_2=mape_loss_2,
    mse_2=mse_loss_2,
    vola=volatility_loss,
    vola_abs=volatility_loss_abs,
    vola_diff=volatility_loss_diff,
    vola_diff_mse=volatility_loss_diff_mse,
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
