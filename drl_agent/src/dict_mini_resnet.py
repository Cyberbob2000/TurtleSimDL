import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from mini_resnet import MinimalResNet

class DictMinimalResNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super(DictMinimalResNet, self).__init__(observation_space, features_dim)
        self.extractors = nn.ModuleDict()
        total_flattened_size = 0

        # create a feature extractor for each key in the observation space
        for key, subspace in observation_space.spaces.items():
            assert isinstance(subspace, spaces.Box), f"Observation space for key {key} must be of type spaces.Box"
            self.extractors[key] = MinimalResNet(subspace, features_dim)
            total_flattened_size += features_dim

        self.final_fc = nn.Sequential(
            nn.Linear(total_flattened_size, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: dict) -> th.Tensor:
        features = []
        for key, extractor in self.extractors.items():
            features.append(extractor(observations[key]))
        concatenated_features = th.cat(features, dim=1)
        return self.final_fc(concatenated_features)