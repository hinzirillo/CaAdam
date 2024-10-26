from caadam import ScalingStrategy
from typing import Dict
from keras import ops

class MultiplicativeMinMaxMedianConnectionScaling(ScalingStrategy):
    def compute_scaling_factor(self, connections: int, gradients, layer_info, model_info: Dict):
        scaling_factor = self.hyperparameters.get('scaling_factor', 5.0)
        if model_info['min_connections'] == model_info['max_connections']:
            return 1.0
        
        # Normalize the connections to a range between -1 and 1
        if connections <= model_info['median_connections']:
            normalized = (model_info['median_connections'] - connections) / (model_info['median_connections'] - model_info['min_connections'])
        else:
            normalized = (connections - model_info['median_connections']) / (model_info['max_connections'] - model_info['median_connections'])
        
        # Use an exponential function to map the normalized value to a scaling factor
        # This will map -1 to 1/scaling_factor, 0 to 1, and 1 to scaling_factor
        return ops.exp(ops.log(scaling_factor) * normalized)

    def get_config(self):
        return {"scaling_factor": self.hyperparameters.get('scaling_factor', 5.0)}

    @classmethod
    def from_config(cls, config):
        return cls(**config)