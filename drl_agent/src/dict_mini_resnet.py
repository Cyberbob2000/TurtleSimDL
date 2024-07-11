import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from mini_resnet import MinimalResNet
import torchvision.models

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

class DictImageNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 32):
        super(DictImageNet, self).__init__(observation_space, features_dim)

        extractors = {}
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "map":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                assert isinstance(subspace, spaces.Box), f"Observation space for key {key} must be of type spaces.Box"
                model_ft = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
                ### strip the last layer
                for param in model_ft.parameters():
                    param.requires_grad = False
                feature_extractor = th.nn.Sequential(*list(model_ft.children())[:-1],nn.Flatten(), nn.Linear(512, features_dim), nn.ReLU(), nn.Flatten())
                extractors[key] = feature_extractor
                #image net without final layer gives 1x512x1x1 output
            elif key == "laser":
                # Flatten vector if needed
                extractors[key] = nn.Flatten()

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually imagenet 32 plus 5laser
        self._features_dim = features_dim+5
    
    def forward(self, observations: dict) -> th.Tensor:
        features = []
        for key, extractor in self.extractors.items():
            if key == "map":
                t =extractor(observations[key])
                features.append(t)
                
            elif key == "laser":
                t2 = extractor(observations[key])
                features.append(t2)
        #Probably need to change to dim = 1 if not only map
        concatenated_features = th.cat(features, dim=1)
        return concatenated_features
