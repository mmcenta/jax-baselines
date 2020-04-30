import functools
import os
import time
from collections import namedtuple

import gym
from gym.spaces import Box, Discrete
import haiku as hk
import jax
import jax.experimental.optix as optix
import jax.numpy as jnp
from jax.experimental.optimizers import adam
from jax import random
import numpy as np

from jax_baselines.common.actor import get_actor
from jax_baselines.common.critic import StateCritic
from jax_baselines.common.learner import ActorLearner, StateCriticLearner
from jax_baselines.common.util import make_preprocessor
from jax_baselines import logger
from jax_baselines.save import load_from_zip, save_to_zip
from jax_baselines.vpg.buffer import VPGBuffer


def learn(rng, env_fn, actor_net_fn, critic_net_fn,
          steps_per_epoch=4000, epochs=50, gamma=0.99, actor_lr=3e-4,
          critic_lr=1e-3, train_value_iters=80, lam=0.97, max_ep_len=1000,
          save_dir='./experiments/vpg', save_freq=10, logger_format_strs=None):
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
    buffer = VPGBuffer(steps_per_epoch, dummy_obs, dummy_act,
                       gamma=gamma, lam=lam)

    # create actor
    def actor_loss_fn(batch, logp):
        adv = batch.advantages
        return -(logp * adv).mean()

    actor = ActorLearner(
        get_actor(action_space, actor_net_fn),
        adam(actor_lr),
        actor_loss_fn,
    )

    # create critc
    def critic_loss_fn(batch, val):
        ret = batch.returns
        return jnp.square(val - ret).mean()

    critic = StateCriticLearner(
        StateCritic(critic_net_fn),
        adam(actor_lr),
        critic_loss_fn,
    )

    # initialize states
    rng, arng, crng = random.split(rng, 3)
    actor_state = actor.init_state(arng, dummy_obs)
    critic_state = critic.init_state(crng, dummy_obs)

    # training loop
    ep_len, ep_ret = 0, 0
    obs = preprocess(env.reset())
    for epoch in range(epochs):
        start_time = time.time()
        ep_lens, ep_rets = [], []

        # experience loop
        for t in range(steps_per_epoch):
            rng, step_rng = random.split(rng)

            # get next action and current state value
            act = actor.step(actor_state, step_rng, obs)
            val = critic.state_value(critic_state, obs).item()

            # convert act to numpy array (because gym doesn't support JAX)
            act = np.array(act)

            # take the action and store the step on the buffer
            next_obs, rew, done, _ = env.step(act)
            epoch_ended = buffer.store(obs, act, rew, val)

            # update episode vars
            obs = preprocess(next_obs)
            ep_len += 1
            ep_ret += rew

            # end episode if necessary
            timeout = (ep_len == max_ep_len)
            terminal = (done or timeout)
            if terminal or epoch_ended:
                if not terminal:
                    print("Warning: episode cut of by epoch {:} at {:} steps.".format(epoch + 1, ep_len + 1))
                # bootstrap last value if not at terminal state
                last_val = 0 if done else critic.state_value(critic_state, obs).item()
                buffer.end_episode(last_val)
                if terminal:
                    ep_lens.append(ep_len)
                    ep_rets.append(ep_ret)
                obs = preprocess(env.reset())
                ep_ret, ep_len = 0, 0

        # save agent
        if (epoch % save_freq == 0) or (epoch == epochs - 1):

            actor_params = actor.get_params(actor_state)
            critic_params = critic.get_params(critic_state)
            agent_dict = {
                'actor_params': actor_params,
                'critic_params': critic_params,
            }
            save_to_zip(save_dir, agent_dict)

        # get batch
        batch = buffer.get_batch()

        # get old logp and losses
        if epoch > 0:
            old_logp = logp
            old_actor_loss = actor_loss
            old_critic_loss = critic_loss
        else:
            old_logp = actor.logp(actor_state, batch.observations, batch.actions)
            old_actor_loss = actor.loss(actor_state, batch)
            old_critic_loss = critic.loss(critic_state, batch)

        # update parameters
        actor_state = actor.update(epoch, actor_state, batch)
        for k in range(train_value_iters):
            critic_state = critic.update(epoch * train_value_iters + k,
                critic_state, batch)

        # get new logp and losses
        logp = actor.logp(actor_state, batch.observations, batch.actions)
        actor_loss = actor.loss(actor_state, batch)
        critic_loss = critic.loss(critic_state, batch)

        # convert to array to extract metrics easily
        ep_lens = jnp.array(ep_lens)
        ep_rets = jnp.array(ep_rets)

        # log epoch
        logger.info('epoch {:}'.format(epoch))

        # log train metrics
        logger.logkv('actor_loss', old_actor_loss)
        logger.logkv('critic_loss', old_critic_loss)
        logger.logkv('kl', (old_logp - logp).sum().item())
        logger.logkv('entropy', (-logp).mean().item())
        logger.logkv('delta_actor_loss', (actor_loss - old_actor_loss))
        logger.logkv('delta_critic_loss', (critic_loss - old_critic_loss))

        # log epoch metrics
        logger.logkv('epoch', epoch)
        logger.logkv('time', time.time() - start_time)
        logger.logkv('ep_len_mean', ep_lens.mean())
        logger.logkv('ep_len_std', ep_lens.std())
        logger.logkv('ep_rets_mean', ep_rets.mean())
        logger.logkv('ep_rets_std', ep_rets.std())
        logger.dumpkvs()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0',
                        help='Environment id.')
    parser.add_argument('--hidden-size', '-hs', type=int, default=64,
                        help='Hidden layer size (same for all layers).')
    parser.add_argument('--layers', '-l', type=int, default=2,
                        help='Number of layers.')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount parameter.')
    parser.add_argument('--lam', type=float, default=0.97,
                        help='Lambda parameter of Generalized Advantage Estimation (GAE).')
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='Seed for the random number generator.')
    parser.add_argument('--steps', type=int, default=4000,
                        help='Number of timesteps per epoch.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs.')
    parser.add_argument('--exp-name', type=str, default='vpg',
                        help='Experiment name for saving.')
    args = parser.parse_args()

    # Path where all experiment data and saves will be dumped
    save_dir = os.path.join('./experiments', args.exp_name)

    # Define actor and critic neural networks
    action_space = gym.make(args.env).action_space
    if isinstance(action_space, Box):
        act_dim = action_space.sample().shape[-1]
    elif isinstance(action_space, Discrete):
        act_dim = action_space.n
    else:
        raise ValueError("Environment action space type not supported.")

    actor_net_fn = lambda obs: hk.nets.MLP(output_sizes=[64, 64, act_dim])(obs)
    critic_net_fn = lambda obs: hk.nets.MLP(output_sizes=[64, 64, 1])(obs)

    # Run experiment
    key = random.PRNGKey(args.seed)
    learn(key, lambda: gym.make(args.env),
          actor_net_fn, critic_net_fn,
          steps_per_epoch=args.steps, epochs=args.epochs, gamma=args.gamma,
          lam=args.lam, save_dir=save_dir)
