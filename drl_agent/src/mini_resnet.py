import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MinimalResNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super(MinimalResNet, self).__init__(observation_space, features_dim)
        assert isinstance(observation_space, spaces.Box), "Observation space must be of type gymnasium.spaces.Box"
        n_input_channels = observation_space.shape[0]  

        # conv layer
        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0)
        self.relu = nn.ReLU()

        # residual block
        self.res_block1 = self._make_res_block(32, 32)
        self.downsample1 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)  

        # residual block 
        self.res_block2 = self._make_res_block(32, 64, downsample=True)
        self.downsample2 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0) 

        self.flatten = nn.Flatten()

        # the number of features
        with th.no_grad():
            n_flatten = self._get_flattened_size(observation_space)

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def _make_res_block(self, in_channels, out_channels, downsample=False):
        stride = 2 if downsample else 1
        res_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        return res_block

    def _get_flattened_size(self, observation_space):
        # determine the size
        sample_input = th.as_tensor(observation_space.sample()[None]).float()
        conv1_out = self.relu(self.conv1(sample_input))

        res_block1_out = self.res_block1(conv1_out)
        res_block1_out += self.downsample1(conv1_out)
        res_block1_out = self.relu(res_block1_out)

        res_block2_out = self.res_block2(res_block1_out)
        res_block2_out += self.downsample2(res_block1_out)
        res_block2_out = self.relu(res_block2_out)

        flattened = self.flatten(res_block2_out)
        return flattened.shape[1]

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.relu(self.conv1(observations))

        res_block1_out = self.res_block1(x)
        res_block1_out += self.downsample1(x)
        x = self.relu(res_block1_out)

        res_block2_out = self.res_block2(x)
        res_block2_out += self.downsample2(x)
        x = self.relu(res_block2_out)

        x = self.flatten(x)
        return self.linear(x)