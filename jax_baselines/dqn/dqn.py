import time

import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental.optimizers import adam
from rlax import huber_loss

from jax_baselines import logger
from jax_baselines.common.critic import DiscreteActionCritic
from jax_baselines.common.learner import ActionCriticLearner
from jax_baselines.common.util import make_preprocessor
from jax_baselines.dqn.replay import ReplayMemory
from jax_baselines.save import load_from_zip, save_to_zip


def learn(rng, env_fn, net_fn,
          gamma=0.99, lr=5e-4, steps=1e6, batch_size=32, epsilon_0=0.1,
          epsilon_scheduler=None, warmup=1000, train_freq=1,
          memory_size=50000,
          save_dir='./experiments/dqn', save_freq=1e4, logger_format_strs=None):
    """
    """

    # configure logger
    logger.configure(dir=save_dir, format_strs=logger_format_strs)
    logger.set_level(logger.INFO)

    # get observation preprocessor
    preprocess = make_preprocessor()

    # get epsilon scheduler
    if epsilon_scheduler is None:
        epsilon_scheduler = lambda i: epsilon_0

    # make sure there are sufficient examples for a batch after warmup
    warmup = max(warmup, batch_size)

    # initialize environment and buffer
    env = env_fn()
    action_space = env.action_space
    dummy_obs = preprocess(env.observation_space.sample())
    dummy_act = env.action_space.sample()
    memory = ReplayMemory(memory_size, dummy_obs, dummy_act)

    # create action critic
    def loss_fn(batch, q_vals, next_q_vals):
        target = batch.rewards + gamma * next_q_vals.max(axis=1)
        return jnp.square(q_vals - next_q_vals).mean()

    critic = ActionCriticLearner(
        DiscreteActionCritic(net_fn, action_space.n),
        adam(lr),
        loss_fn,
    )

    # initialize state
    rng, init_rng = random.split(rng)
    state = critic.init_state(init_rng, dummy_obs)

    # training loop
    start_time = time.time()
    obs = preprocess(env.reset())
    for i in range(steps):
        rng, step_rng = random.split(rng)

        # take action
        epsilon = epsilon_scheduler(i)
        if random.uniform(step_rng) < epsilon:
            act = action_space.sample()
        else:
            q_vals = critic.action_values(state, obs)
            act = jnp.argmax(q_vals).item()
        new_obs, rew, done, _ = env.step(act)

        # store transition and update obs
        new_obs = preprocess(new_obs)
        memory.store(obs, act, rew, new_obs)

        # start new episode if finished
        if done:
            obs = preprocess(env.reset())
        else:
            obs = new_obs

        # update the network
        if i > warmup and i % train_freq == 0:
            j = (i - warmup) // train_freq # update step
            batch = memory.get_batch(batch_size)

            # get old loss
            if j > 0:
                old_loss = loss
            else:
                old_loss = critic.loss(state, batch)

            # update paramaters
            state = critic.update(j, state, batch)

            # get loss
            loss = critic.loss(state, batch)

            # log update metrics
            logger.info('Update {:}'.format(j))
            logger.logkv('loss', old_loss)
            logger.logkv('delta_loss', loss - old_loss)
            logger.dumpkvs()

        # save model
        if i % save_freq == 0:
            params = critic.get_params(state)
            save_to_zip(dict(params=params))


if __name__ == "__main__":
    import argparse
    pass