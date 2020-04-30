import haiku as hk
import jax
import jax.numpy as jnp
from jax.experimental.optimizers import adam
from rlax import huber_loss

from jax_baselines import logger
from jax_baselines.common.critic import DiscreteActionCritic
from jax_baselines.common.learner import ActionCriticLearner
from jax_baselines.common.util import make_preprocessor
from jax_baselines.dqn.replay import ReplayMemory


def learn(env_fn, net_fn,
          gamma=0.99, lr=5e-4, steps=1e6, batch_size=32, epsilon=0.1,
          epsilon_schedule=None, learning_starts=1000, train_freq=1,
          memory_size=50000, frame_skip=1,
          save_dir='./experiments/dqn', save_freq=1e4, logger_format_strs=None):
    """
    """

    # configure logger
    logger.configure(dir=save_dir, format_strs=logger_format_strs)
    logger.set_level(logger.INFO)

    # get observation preprocessor
    preprocess = make_preprocessor()

    # initialize environment and buffer
    env = env_fn()
    action_space = env.action_space
    dummy_obs = preprocess(env.observation_space.sample())
    dummy_act = env.action_space.sample()
    memory = ReplayMemory(memory_size, dummy_obs, dummy_act)

    # create action critic
    def loss_fn(batch, action_vals):
        obs, act, rew, new_obs = batch.observations, batch.actions, batch.rewards, batch.new_observation
        

    critic = ActionCriticLearner(
        DiscreteActionCritic(net_fn, action_space.n),
        adam(lr),
        loss_fn,
    )
