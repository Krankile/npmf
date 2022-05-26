from typing import List

import numpy as np


class EarlyStop:
    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.triggers = 0

    def __call__(self, epoch_loss: List[float]) -> bool:
        loss = np.mean(epoch_loss)

        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.triggers = 0
        else:
            self.triggers += 1

        return not self.triggers < self.patience
