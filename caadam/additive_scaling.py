from caadam import ScalingStrategy
from typing import Dict

class AdditiveMinMaxMedianConnectionScaling(ScalingStrategy):
    def compute_scaling_factor(self, connections: int, gradients, layer_info, model_info: Dict):
        scaling_factor = self.hyperparameters.get('scaling_factor', 0.95)
        if model_info['min_connections'] == model_info['max_connections']:
            return 1.0
        if connections <= model_info['median_connections']:
            return 1 + scaling_factor * (
                (model_info['median_connections'] - connections) / 
                (model_info['median_connections'] - model_info['min_connections'])
            )
        else:
            return 1 - scaling_factor * (
                (connections - model_info['median_connections']) / 
                (model_info['max_connections'] - model_info['median_connections'])
            )
            
    def get_config(self):
        return {"scaling_factor": self.hyperparameters.get('scaling_factor', 0.95)}

    @classmethod
    def from_config(cls, config):
        return cls(**config)