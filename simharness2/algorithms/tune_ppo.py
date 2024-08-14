from ray.rllib.algorithms import PPO, Algorithm, AlgorithmConfig
from ray.rllib.utils.annotations import override


class TunePPO(PPO):
    @override(Algorithm)
    def setup(self, config: AlgorithmConfig, simharness_config=None) -> None:
        breakpoint()
        # Add custom setup code here.
        super().setup(config)
