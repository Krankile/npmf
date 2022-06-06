from typing import Dict
from .naive_model import naive_models
from .tcn_model import tcn_models
from .naive_tcn_model import naive_tcn_models

from torch import nn

models: Dict[str, nn.Module] = {**naive_models, **tcn_models, **naive_tcn_models}

__all__ = [models]
