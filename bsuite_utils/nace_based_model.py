import copy
import sys
import time
import random
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, ClassVar
from abc import ABC, abstractmethod
from collections import deque
from torch.nn import functional as F

import numpy as np
import torch as th
from gymnasium import spaces
import gymnasium as gymnasium
import nace

from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, ProgressBarCallback
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, \
    MultiInputActorCriticPolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.patch_gym import _convert_space, _patch_env
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common import utils
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
    unwrap_vec_normalize,
)
from stable_baselines3.common.utils import (
    check_for_correct_spaces,
    get_device,
    get_schedule_fn,
    get_system_info,
    set_random_seed,
    update_learning_rate,
)

from stable_baselines3.common.preprocessing import check_for_nested_spaces, is_image_space, \
    is_image_space_channels_first

SelfOnPolicyAlgorithm = TypeVar("SelfNaceAlgorithm", bound="NaceAlgorithm")


def maybe_make_env(env: Union[GymEnv, str], verbose: int) -> GymEnv:
    """If env is a string, make the environment; otherwise, return env.

    :param env: The environment to learn from.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating if envrironment is created
    :return A Gym (vector) environment.
    """
    if isinstance(env, str):
        env_id = env
        if verbose >= 1:
            print(f"Creating environment from the given name '{env_id}'")
        # Set render_mode to `rgb_array` as default, so we can record video
        try:
            env = gymnasium.make(env_id, render_mode="rgb_array")
        except TypeError:
            env = gymnasium.make(env_id)
    return env


class NaceAlgorithm(ABC):
    rollout_buffer: RolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            # learning_rate: Union[float, Schedule],
            # n_steps: int,
            # gamma: float,
            # gae_lambda: float,
            # ent_coef: float,
            # vf_coef: float,
            # max_grad_norm: float,
            # use_sde: bool,
            # sde_sample_freq: int,
            # rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
            # rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
            # stats_window_size: int = 100,
            # tensorboard_log: Optional[str] = None,
            # monitor_wrapper: bool = True,
            # policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "mps",
            monitor_wrapper: bool = True,
            # When creating an environment, whether to wrap it or not in a Monitor wrapper
            _init_setup_model: bool = True,
            supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
    ):
        self.policy = policy
        self.env = env
        self.verbose = verbose
        self.device = device
        self.support_multi_env = True
        self.seed = seed
        self.supported_action_spaces = supported_action_spaces
        self.rollout_buffer_class = None
        self.use_sde = False  # not used/implemented use generalized State Dependent Exploration (gSDE) over action noise exploration
        self.n_steps = 5  # n_steps: The number of steps to run for each environment per update
        # (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
        self.gamma = 0.0  # not used/implemented
        self.sde_sample_freq = 0  # not used/implemented
        self.gae_lambda = 0.0  # not used/implemented
        self.rollout_buffer_kwargs = {}  # not used/implemented
        self.policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
            "MlpPolicy": ActorCriticPolicy,
            "CnnPolicy": ActorCriticCnnPolicy,
            "MultiInputPolicy": MultiInputActorCriticPolicy,
        }
        self.learning_rate = 0.0  # not used/implemented
        self.policy_kwargs = {}  # not used/implemented
        self.lr_schedule = get_schedule_fn(self.learning_rate)
        # Buffers for logging
        self._stats_window_size = 100
        self.ep_info_buffer = None  # type: Optional[deque]
        self.ep_success_buffer = None  # type: Optional[deque]
        #
        self.action_noise: Optional[ActionNoise] = None
        self.num_timesteps = 0
        self._last_obs = None  # type: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
        self._custom_logger = False
        self.tensorboard_log: Optional[str] = None  # location of tensor board logging
        self._n_updates: int = 0

        if isinstance(policy, str):
            self.policy_class = self._get_policy_from_name(policy)
        else:
            self.policy_class = policy

        # Create and wrap the env if needed
        if env is not None:
            env = maybe_make_env(env, self.verbose)
            env = self._wrap_env(env, self.verbose, monitor_wrapper)

            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.n_envs = env.num_envs
            self.env = env

            # get VecNormalize object if needed
            self._vec_normalize_env = unwrap_vec_normalize(env)

            if supported_action_spaces is not None:
                assert isinstance(self.action_space, supported_action_spaces), (
                    f"The algorithm only supports {supported_action_spaces} as action spaces "
                    f"but {self.action_space} was provided"
                )

            if self.n_envs > 1:
                raise ValueError(
                    "Error: the model does not support multiple envs; it requires " "a single vectorized environment."
                )

            # Catch common mistake: using MlpPolicy/CnnPolicy instead of MultiInputPolicy
            if policy in ["MlpPolicy", "CnnPolicy"] and isinstance(self.observation_space, spaces.Dict):
                raise ValueError(
                    f"You must use `MultiInputPolicy` when working with dict observation space, not {policy}")

            if self.use_sde and not isinstance(self.action_space, spaces.Box):
                raise ValueError(
                    "generalized State-Dependent Exploration (gSDE) can only be used with continuous actions.")

            if isinstance(self.action_space, spaces.Box):
                assert np.all(
                    np.isfinite(np.array([self.action_space.low, self.action_space.high]))
                ), "Continuous action space must have a finite lower and upper bound"



        if _init_setup_model:
            self._setup_model()





    def _update_info_buffer(self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None) -> None:
        """

        copied from base algorthm
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.

        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """

        copied from base algorthm

        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    @property
    def logger(self) -> Logger:
        # copied from BaseAlgorthm
        """Getter for the logger object."""
        return self._logger

    def _init_callback(
            self,
            callback: MaybeCallback,
            progress_bar: bool = False,
    ) -> BaseCallback:
        """

        copied from BaseAlgorythm
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: A hybrid callback calling `callback` and performing evaluation.
        """
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        # Add progress bar callback
        if progress_bar:
            callback = CallbackList([callback, ProgressBarCallback()])

        callback.init_callback(self)
        return callback

    def _setup_learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            reset_num_timesteps: bool = True,
            tb_log_name: str = "run",
            progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        """
        Copied from BaseAlgorithm

        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: Total timesteps and callback(s)
        """
        self.start_time = time.time_ns()

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=self._stats_window_size)
            self.ep_success_buffer = deque(maxlen=self._stats_window_size)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            assert self.env is not None
            self._last_obs = self.env.reset()  # type: ignore[assignment]
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()

        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        # Create eval callback if needed
        callback = self._init_callback(callback, progress_bar)

        return total_timesteps, callback

    def _get_policy_from_name(self, policy_name: str) -> Type[BasePolicy]:
        """
        Get a policy class from its name representation.

        The goal here is to standardize policy naming, e.g.
        all algorithms can call upon "MlpPolicy" or "CnnPolicy",
        and they receive respective policies that work for them.

        :param policy_name: Alias of the policy
        :return: A policy class (type)
        """

        if policy_name in self.policy_aliases:
            return self.policy_aliases[policy_name]
        else:
            raise ValueError(f"Policy {policy_name} unknown")

    @staticmethod
    def _wrap_env(env: GymEnv, verbose: int = 0, monitor_wrapper: bool = True) -> VecEnv:
        """ "
        Wrap environment with the appropriate wrappers if needed.
        For instance, to have a vectorized environment
        or to re-order the image channels.

        :param env:
        :param verbose: Verbosity level: 0 for no output, 1 for indicating wrappers used
        :param monitor_wrapper: Whether to wrap the env in a ``Monitor`` when possible.
        :return: The wrapped environment.
        """
        if not isinstance(env, VecEnv):
            # Patch to support gym 0.21/0.26 and gymnasium
            env = _patch_env(env)
            if not is_wrapped(env, Monitor) and monitor_wrapper:
                if verbose >= 1:
                    print("Wrapping the env with a `Monitor` wrapper")
                env = Monitor(env)
            if verbose >= 1:
                print("Wrapping the env in a DummyVecEnv.")
            env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

        # Make sure that dict-spaces are not nested (not supported)
        check_for_nested_spaces(env.observation_space)

        if not is_vecenv_wrapped(env, VecTransposeImage):
            wrap_with_vectranspose = False
            if isinstance(env.observation_space, spaces.Dict):
                # If even one of the keys is a image-space in need of transpose, apply transpose
                # If the image spaces are not consistent (for instance one is channel first,
                # the other channel last), VecTransposeImage will throw an error
                for space in env.observation_space.spaces.values():
                    wrap_with_vectranspose = wrap_with_vectranspose or (
                            is_image_space(space) and not is_image_space_channels_first(space)  # type: ignore[arg-type]
                    )
            else:
                wrap_with_vectranspose = is_image_space(env.observation_space) and not is_image_space_channels_first(
                    env.observation_space  # type: ignore[arg-type]
                )

            if wrap_with_vectranspose:
                if verbose >= 1:
                    print("Wrapping the env in a VecTransposeImage.")
                env = VecTransposeImage(env)

        return env

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)

        :param seed:
        """
        if seed is None:
            return

        """
        Seed the different random generators.

        :param seed:
        :param using_cuda:
        """
        # Seed python RNG
        random.seed(seed)
        # Seed numpy RNG
        np.random.seed(seed)
        # seed the RNG for all devices (both CPU and CUDA)
        th.manual_seed(seed)

        if self.device.type == th.device("cuda").type:  # using_cuda
            # Deterministic operations for CuDNN, it may impact performances
            th.backends.cudnn.deterministic = True
            th.backends.cudnn.benchmark = False

        self.action_space.seed(seed)
        # self.env is always a VecEnv
        if self.env is not None:
            self.env.seed(seed)

    def _setup_model(self) -> None:
        self.set_random_seed(self.seed)

        # if self.rollout_buffer_class is None:
        #     if isinstance(self.observation_space, spaces.Dict):
        #         self.rollout_buffer_class = DictRolloutBuffer
        #     else:
        #         self.rollout_buffer_class = RolloutBuffer
        #
        # self.rollout_buffer = self.rollout_buffer_class(
        #     self.n_steps,
        #     self.observation_space,  # type: ignore[arg-type]
        #     self.action_space,
        #     device=self.device,
        #     gamma=self.gamma,
        #     gae_lambda=self.gae_lambda,
        #     n_envs=self.n_envs,
        #     **self.rollout_buffer_kwargs,
        # )
        # self.policy = self.policy_class(  # type: ignore[assignment]
        #     self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        # )
        # self.policy = self.policy.to(self.device)

        self.stepper = nace.stepper_v4.StepperV4(agent_indication_value=1)
        self.full_view_npworld = None
        self.time_counter = 0

        nace.world_module.set_traversable_board_value(chr(0)) # set '0' to be traversable (should be learnt? or not needed)

        if isinstance(self.action_space, spaces.Discrete):
            numeric_action_list = [ i+self.action_space.start for i in range(self.action_space.n)]
            fnc_action_list = [nace.world_module.left, nace.world_module.right]
            nace.world_module.set_full_action_list(fnc_action_list)
            self.action_lookup = {}
            for (fnc, numeric) in zip( fnc_action_list, numeric_action_list):
                self.action_lookup[fnc] = numeric
        else:
            # set the mapping of the movements, the rest are expected to be learnt. (these could be learnt from watching gym
            # action and this and last worlds.)
            print("ERROR: This line should not be logged or used, this path should never trigger. TODO delete") # TODO delete
            nace.world_module.set_full_action_list(
                [nace.world_module.up, nace.world_module.right, nace.world_module.down, nace.world_module.left])

        print("TODO determine if the next line is needed, or needed to be refactored in some way") # TODO
        nace.hypothesis.Hypothesis_UseMovementOpAssumptions(
            nace.world_module.left,
            nace.world_module.right,
            nace.world_module.up,
            nace.world_module.down,
            nace.world_module.drop,
            "DisableOpSymmetryAssumption" in sys.argv,
        )




    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
    ) -> bool:
        """
        # Based on the code in on_policy_algorithm.py collect_rollouts()


        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
                # actions tensor shape (1,) int64
                # values tensor shape (1,1) float32
                # logprob tensor shape (1,)  float32
                # Of course we need to add the observations times in.

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """

        """
        # copied from A2C

        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        # self._update_learning_rate(self.policy.optimizer)

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            # Normalize advantage (not present in the original implementation)
            advantages = rollout_data.advantages
            # if self.normalize_advantage:
            #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss
            policy_loss = -(advantages * log_prob).mean()

            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values)

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            # loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
            loss = policy_loss + entropy_loss + value_loss

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            # th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    def _dump_logs(self, iteration: int) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    def _learn_original(
            self: SelfOnPolicyAlgorithm,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            tb_log_name: str = "OnPolicyAlgorithm",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        # copied from on_policy_algorithm.py

        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            self.train()  #

        callback.on_training_end()

        return self

    def _new_train_and_rollout(
            self,
            env: VecEnv,
            callback: BaseCallback,
            n_rollout_steps: int,
    ) -> bool:
        """
        # Based on the code in on_policy_algorithm.py collect_rollouts()


        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts (may be consumed imediately! :)
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        # self.policy.set_training_mode(False)

        n_steps = 0
        # Sample new weights for the state dependent exploration
        # if self.use_sde:
        #     self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        accumulated_rewards = None

        while n_steps < n_rollout_steps:
            # if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
            #     # Sample a new noise matrix
            #     self.policy.reset_noise(env.num_envs)

            # with th.no_grad():
            #     # Convert to pytorch tensor or to TensorDict
            #     # obs_tensor = obs_as_tensor(self._last_obs, self.device)
            #     # actions_old, values, log_probs = self.policy(obs_tensor)  # TODO replace this call
            #     # actions tensor shape (1,) int64
            #     # values tensor shape (1,1) float32
            #     # logprob tensor shape (1,)  float32

            if self.full_view_npworld is None:
                print("NACE: creating new world.")
                self.full_view_npworld = nace.world_module_numpy.NPWorld(
                    with_observed_time=False,
                    name="external_npworld",
                    view_dist_x=100,
                    view_dist_y=100)

            # copy the env into a known foram world (this step could be optimised out)
            agent_xy_loc_list, modified_count, pre_update_world = self.full_view_npworld.update_world_from_ground_truth_NPArray(
                observed_word=self._last_obs[0]
            )

            action, current_behavior = self.stepper.get_next_action(
                ground_truth_external_world=self.full_view_npworld,
                new_xy_loc=agent_xy_loc_list[-1],
                print_debug_info=True
            )
            # _x = env.render("human")

            actions = np.zeros( (1,) )
            actions[0] = self.action_lookup[action]

                # Of course we need to add the observations times in.

            # Rescale and perform action
            clipped_actions = actions

            # if isinstance(self.action_space, spaces.Box):
            #     if self.policy.squash_output:
            #         # Unscale the actions to match env bounds
            #         # if they were previously squashed (scaled in [-1, 1])
            #         clipped_actions = self.policy.unscale_action(clipped_actions)
            #     else:
            #         # Otherwise, clip the actions to avoid out of bound error
            #         # as we are sampling from an unbounded Gaussian distribution
            #         clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            # done[0] == True on timestep 19 on 20x20 board
            new_obs, rewards, dones, infos = env.step(clipped_actions)

            if accumulated_rewards is None:
                accumulated_rewards = copy.deepcopy(rewards)
            else:
                accumulated_rewards += rewards

            # copy state from env format into NPformat
            agent_xy_loc_list, modified_count, pre_update_world = self.full_view_npworld.update_world_from_ground_truth_NPArray(
                observed_word=new_obs[0]
            )
            # let stepper update it's internal world state
            self.stepper.set_world_ground_truth_state(self.full_view_npworld, agent_xy_loc_list, self.time_counter)
            self.time_counter += 1

            # let stepper get the latest agent state
            status = self.stepper.set_agent_ground_truth_state(
                xy_loc=agent_xy_loc_list[-1],
                score=accumulated_rewards[0],
                values_exc_score=[] # no state held by this agent (i.e. keys, money)
            )
            # perform learning
            self.stepper.predict_and_observe(print_out_world_and_plan=True)


            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        # with th.no_grad():
        #     # Compute value for the last timestep
        #     values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        # rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def _learn_new(
            self: SelfOnPolicyAlgorithm,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            tb_log_name: str = "OnPolicyAlgorithm",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        # copied from on_policy_algorithm.py

        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:

            # TODO change this block.
            continue_training = self._new_train_and_rollout(
                self.env,
                callback,
                n_rollout_steps=self.n_steps)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

        callback.on_training_end()

        return self

    def learn(
            self: SelfOnPolicyAlgorithm,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            tb_log_name: str = "OnPolicyAlgorithm",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        # call our new, or  the old original impl
        return self._learn_new(total_timesteps,
                               callback,
                               log_interval,
                               tb_log_name,
                               reset_num_timesteps,
                               progress_bar)

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []


