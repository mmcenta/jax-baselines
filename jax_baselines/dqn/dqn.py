import os
import time

import gym
from gym.spaces import Discrete
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental.optimizers import adam
import numpy as np
from rlax import huber_loss

from jax_baselines import logger
from jax_baselines.common.critic import DiscreteActionCritic
from jax_baselines.common.learner import ActionCriticLearner
from jax_baselines.common.scheduler import LinearDecay
from jax_baselines.common.util import make_preprocessor
from jax_baselines.dqn.replay import ReplayMemory
from jax_baselines.save import load_from_zip, save_to_zip


def learn(rng, env_fn, net_fn,
          gamma=0.99, lr=5e-4, steps=1e6, batch_size=100, epsilon=0.1,
          epsilon_scheduler=None, warmup=1000, train_freq=1,
          memory_size=50000, eval_freq=1e4, eval_episodes=10,
          save_dir='./experiments/dqn', save_freq=1e4,
          logger_format_strs=None):
    """
    """

    # configure logger
    logger.configure(dir=save_dir, format_strs=logger_format_strs)
    logger.set_level(logger.INFO)

    # get observation preprocessor
    preprocess = make_preprocessor()

    # get epsilon scheduler
    if epsilon_scheduler is None:
        epsilon_scheduler = lambda i: epsilon

    # make sure there are sufficient examples for a batch after warmup
    warmup = max(warmup, batch_size)

    # initialize environment and buffer
    env = env_fn()
    action_space = env.action_space
    if not isinstance(action_space, Discrete):
        raise ValueError('Environment action space must be discrete.')

    # initialize replay memory
    dummy_obs = preprocess(env.observation_space.sample())
    dummy_act = env.action_space.sample()
    memory = ReplayMemory(memory_size, dummy_obs, dummy_act)

    # create action critic
    def loss_fn(batch, q_val, next_q_vals):
        target = batch.rewards + gamma * next_q_vals.max(axis=1)
        return huber_loss(q_val -  target).mean()

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
    loss_mean = 0
    obs = preprocess(env.reset())
    for i in range(int(steps)):
        rng, step_rng = random.split(rng)

        # take action (epsilon-greedy approach)
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
        if i >= warmup and (i - warmup) % train_freq == 0:
            j = (i - warmup) // train_freq # update step
            batch = memory.get_batch(batch_size)

            # update paramaters
            state = critic.update(j, state, batch)

            # get loss
            loss = critic.loss(state, batch)
            loss_mean += loss / eval_freq

        # run evaluation
        if i % eval_freq == 0 and i > 0:
            ep_lens, ep_rets = [], []
            qs = [] # store the chosen action q value
            eval_env = env_fn()

            for _ in range(eval_episodes):
                ep_ret, ep_len = 0, 0
                done = False
                obs = preprocess(eval_env.reset())
                while not done:
                    q_vals = critic.action_values(state, obs)
                    qs.append(jnp.max(q_vals).item())
                    act = jnp.argmax(q_vals).item()
                    obs, rew, done, _ = eval_env.step(act)
                    obs = preprocess(obs)
                    ep_len += 1
                    ep_ret += rew
                ep_lens.append(ep_len)
                ep_rets.append(ep_ret)

            # log results
            ep_lens, ep_rets = np.array(ep_lens), np.array(ep_rets)
            logger.info('evaluation at step {:}'.format(i))
            logger.info('elapsed time {:}'.format(time.time() - start_time))
            logger.logkv('loss_mean', loss_mean)
            logger.logkv('ep_len_mean', ep_lens.mean())
            logger.logkv('ep_len_std', ep_lens.std())
            logger.logkv('ep_ret_mean', ep_rets.mean())
            logger.logkv('ep_ret_std', ep_rets.std())
            logger.logkv('q_mean', sum(q_vals) / len(q_vals))
            logger.dumpkvs()


        # save model
        if i % save_freq == 0:
            params = critic.get_params(state)
            save_to_zip(save_dir, dict(params=params))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0',
                        help='Environment id.')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount parameter.')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate.')
    parser.add_argument('--initial-epsilon', type=float, default=0.1,
                        help='Initial value for epsilon.')
    parser.add_argument('--final-epsilon', type=float, default=0.01,
                        help='Final value of epsilon.')
    parser.add_argument('--decay-proportion', type=float, default=0.2,
                        help='Proportion of the training in which epsilon is annealed.')
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='Seed for the random number generator.')
    parser.add_argument('--steps', type=int, default=1e6,
                        help='Number of training timesteps.')
    parser.add_argument('--eval-freq', type=int, default=1e4,
                        help='Number of timesteps between evaluations.')
    parser.add_argument('--warmup', type=int, default=1000,
                        help='Number of timesteps before learning.')
    parser.add_argument('--exp-name', type=str, default='dqn',
                        help='Experiment name for saving.')
    args = parser.parse_args()

    # Path where all experiment data and saves will be dumped
    save_dir = os.path.join('./experiments', args.exp_name)

    # Define Q-network (without output layer)
    net_fn = lambda obs: hk.nets.MLP(output_sizes=[32, 32])(obs)

    # Create learning rate scheduler
    epsilon_scheduler = LinearDecay(
        args.initial_epsilon, args.final_epsilon, args.decay_proportion * args.steps)

    # Run experiment
    key = random.PRNGKey(args.seed)
    learn(key, lambda: gym.make(args.env), net_fn,
        gamma=args.gamma, lr=args.lr, steps=args.steps,
        eval_freq=args.eval_freq, warmup=args.warmup,
        epsilon_scheduler=epsilon_scheduler, save_dir=save_dir)