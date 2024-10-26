import numpy as np
from keras import ops
from keras.optimizers import Optimizer
from caadam import ScalingStrategy
from caadam import AdditiveMinMaxMedianConnectionScaling, MultiplicativeMinMaxMedianConnectionScaling, DepthConnectionScaling


class ConnectionAwareAdam(Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        scaling_strategy: ScalingStrategy=AdditiveMinMaxMedianConnectionScaling(scaling_factor=0.9),
        name="ConnectionAwareAdam",
        **kwargs
    ):
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        if not isinstance(scaling_strategy, ScalingStrategy):
            raise ValueError("scaling_strategy must be an instance of ScalingStrategy")
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.scaling_strategy = scaling_strategy
        self.connection_counts = {}
        self.min_connections = float('inf')
        self.max_connections = 0
        self.median_connections = 0
        self.model_info = {}

    def build(self, var_list):
        super().build(var_list)
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(reference_variable=var, name="m")
            )
            self._velocities.append(
                self.add_variable_from_reference(reference_variable=var, name="v")
            )
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(reference_variable=var, name="vhat")
                )
        self._calculate_connection_counts(var_list)

    def _calculate_connection_counts(self, var_list):
        all_connections = []
        for var in var_list:
            if len(var.shape) > 1:  # Only consider variables with more than 1 dimension (i.e., not biases)
                connections = np.prod(var.shape)
                self.connection_counts[var.name] = connections
                all_connections.append(connections)

        if all_connections:
            self.min_connections = min(all_connections)
            self.max_connections = max(all_connections)
            self.median_connections = np.median(all_connections)
        else:
            self.min_connections = self.max_connections = self.median_connections = 1  # Default to avoid division by zero

        self.model_info = {
            'min_connections': self.min_connections,
            'max_connections': self.max_connections,
            'median_connections': self.median_connections,
            'total_depth': len(var_list),
        }

    def update_step(self, gradient, variable, learning_rate):
        lr = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        local_step = ops.cast(self.iterations + 1, variable.dtype)
        beta_1_power = ops.power(ops.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = ops.power(ops.cast(self.beta_2, variable.dtype), local_step)

        m = self._momentums[self._get_variable_index(variable)]
        v = self._velocities[self._get_variable_index(variable)]

        # Apply scaling strategy
        connections = self.connection_counts.get(variable.name, 0)
        layer_info = {
            'depth': self._get_variable_index(variable),
        }
        scaling_factor = self.scaling_strategy.compute_scaling_factor(
            connections, gradient, layer_info, self.model_info
        )

        alpha = lr * ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)
        alpha *= ops.cast(scaling_factor, variable.dtype)

        m.assign(self.beta_1 * m + (1 - self.beta_1) * gradient)
        v.assign(self.beta_2 * v + (1 - self.beta_2) * ops.square(gradient))

        if self.amsgrad:
            v_hat = self._velocity_hats[self._get_variable_index(variable)]
            v_hat.assign(ops.maximum(v_hat, v))
            v = v_hat

        variable.assign_sub(alpha * m / (ops.sqrt(v) + self.epsilon))

    def get_config(self):
        config = super().get_config()
        config.update({
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "epsilon": self.epsilon,
            "amsgrad": self.amsgrad,
            "scaling_strategy": {
                "class_name": self.scaling_strategy.__class__.__name__,
                "config": self.scaling_strategy.get_config()
            }
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Extract scaling strategy config
        scaling_strategy_config = config.pop("scaling_strategy")
        scaling_strategy_class = globals()[scaling_strategy_config["class_name"]]
        scaling_strategy = scaling_strategy_class.from_config(
            scaling_strategy_config["config"]
        )
        
        # Create optimizer instance
        return cls(scaling_strategy=scaling_strategy, **config)