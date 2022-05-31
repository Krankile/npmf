from typing import Dict
from .naive_model import naive_models
from .tcn_model import tcn_models

from torch import nn

models: Dict[str, nn.Module] = {**naive_models, **tcn_models}

__all__ = [models]
