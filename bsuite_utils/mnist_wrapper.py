import gym
import numpy as np
from gym import spaces
from gym.spaces import Discrete
from skimage import transform
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn import CnnPolicy
import torch
from torch import nn

X = 28  # 36 if using NatureCNN

_observation_space = spaces.Box(low=0, high=255, shape=(1, X, X), dtype=np.uint8)

def obs2img(img):
    # Uncomment if using NatureCNN
    # img = transform.resize(img, (X, X), anti_aliasing=True)
    return img.reshape((1, X, X)).astype(np.uint8)


class MNISTWrapper(gym.Env):
    # Stable baselines v3 has strict requirements for what counts as an image.
    # It must have dtype=np.uint8, 3 dimensions, and the channel axis must be
    # in {1,3} for grayscale, rgb respectively.
    # Channel first: (C, H, W), and minimum resolution is 36x36
    def __init__(self, env: gym.Env):
        self._env = env

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return obs2img(obs), reward, done, info

    def reset(self):
        obs = self._env.reset()
        return obs2img(obs)

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return _observation_space

    def render(self, mode="human"):
        pass


class SmallCNN(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: spaces.Box,
            features_dim: int = 512,  # I don't think this matters
            normalized_image: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 24, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=5, stride=1, padding='same'),
            # nn.ReLU(),
            # nn.Conv2d(48, 48, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class CustomCNNPolicy(CnnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, features_extractor_class=SmallCNN)
