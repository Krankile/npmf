from .naive_model import naive_models
from .tcn_model import tcn_models


models = {**naive_models, **tcn_models}

__all__ = [models]
