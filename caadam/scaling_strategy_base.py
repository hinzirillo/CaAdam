from abc import ABC, abstractmethod
from typing import Dict

class ScalingStrategy(ABC):
    def __init__(self, **kwargs):
        self.hyperparameters = kwargs

    @abstractmethod
    def compute_scaling_factor(self, connections, gradients, layer_info, model_info: Dict):
        raise NotImplementedError

    @abstractmethod
    def get_config(self):
        """Return configuration of the scaling strategy."""
        raise NotImplementedError
    
    @classmethod
    @abstractmethod
    def from_config(cls, config):
        """Create scaling strategy from configuration."""
        raise NotImplementedError