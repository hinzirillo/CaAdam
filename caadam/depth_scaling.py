from caadam import ScalingStrategy
from typing import Dict


class DepthConnectionScaling(ScalingStrategy):
    def compute_scaling_factor(self, connections: int, gradients, layer_info, model_info: Dict):
        scaling_factor = self.hyperparameters.get('scaling_factor', 1.0)
        return (1. + scaling_factor) ** ((model_info['total_depth'] - (1 + layer_info['depth'])) / model_info['total_depth'])

    def get_config(self):
        return {"scaling_factor": self.hyperparameters.get('scaling_factor', 1.0)}

    @classmethod
    def from_config(cls, config):
        return cls(scaling_factor=config['scaling_factor'])
        