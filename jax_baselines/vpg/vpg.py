import os
import time

import flax
from flax import nn
from flax import optim
import gym
from gym.spaces import Box, Discrete
import jax
import jax.numpy as jnp
from jax import random
from jax.ops import index_update

from jax_baselines.common.functions import reverse_discount_cumsum
from jax_baselines.common.modules import MLPCategoricalActor, MLPGaussianActor, MLPCritic
from jax_baselines.common.util import get_shape, add_batch_dim
from jax_baselines import logger
from jax_baselines.save import load_from_zip, save_to_zip


class VPGBuffer:
    """
    """
    def __init__(self, batch_size, obs_dim, act_dim,
                 gamma=0.99, lam=0.95, eps=1e-8):
        """
        """
        self.obs_buf = jnp.zeros(shape=add_batch_dim(batch_size, obs_dim))
        self.act_buf = jnp.zeros(shape=add_batch_dim(batch_size, obs_dim))
        self.rew_buf = jnp.zeros(shape=(batch_size,))
        self.ret_buf = jnp.zeros(shape=(batch_size,))
        self.val_buf = jnp.zeros(shape=(batch_size,))
        self.logp_buf = jnp.zeros(shape=(batch_size,))
        self.adv_buf = jnp.zeros(shape=(batch_size,))
        self.gamma, self.lam, self.eps = gamma, lam, eps
        self.ptr, self.trajectory_start, self.batch_size = 0, 0, batch_size

    def store_timestep(self, obs, act, rew, val, logp):
        """
        """
        assert self.ptr < self.batch_size, "VPGBuffer is full"
        self.obs_buf = index_update(self.obs_buf, self.ptr, obs)
        self.act_buf = index_update(self.act_buf, self.ptr, act)
        self.rew_buf = index_update(self.rew_buf, self.ptr, rew)
        self.val_buf = index_update(self.val_buf, self.ptr, val)
        self.logp_buf = index_update(self.logp_buf, self.ptr, logp)
        self.ptr += 1
        return self.ptr == self.batch_size

    def end_trajectory(self, last_val=0):
        """
        """
        trajectory = jax.ops.index[self.trajectory_start:self.ptr]

        # get the rewards and values of the trajectory
        rews = jnp.append(self.rew_buf[trajectory], last_val)
        vals = jnp.append(self.val_buf[trajectory], last_val)

        # estimate the advantege using the GAE method
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv = reverse_discount_cumsum(deltas, self.gamma * self.lam)
        self.adv_buf = index_update(self.adv_buf, trajectory, adv)

        # compute the rewards-to-go
        ret = reverse_discount_cumsum(rews, self.gamma)[:-1]
        self.ret_buf = index_update(self.ret_buf, trajectory, ret)

        self.trajectory_start = self.ptr

    def get_batch(self):
        """
        """
        assert self.ptr == self.batch_size, "VPGBuffer batch not complete"
        self.ptr, self.trajectory_start = 0, 0

        # normalize the advantages
        adv_mean, adv_std = jnp.mean(self.adv_buf), jnp.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + self.eps)
        return (
            jax.device_put(self.obs_buf),
            jax.device_put(self.act_buf),
            jax.device_put(self.adv_buf),
            jax.device_put(self.ret_buf),
            jax.device_put(self.logp_buf),
        )


def learn(rng, env_fn, actor_def, critic_def,
          steps_per_epoch=4000, epochs=50, gamma=0.99, actor_lr=3e-4,
          critic_lr=1e-3, train_value_iters=80, lam=0.97, max_ep_len=1000,
          save_dir='./experiments/vpg', save_freq=10, logger_format_strs=None):
    """
    """

    # Configure logger
    logger.configure(dir=save_dir, format_strs=logger_format_strs)

    # initialize environment and buffer
    env = env_fn()
    action_space = env.action_space
    obs_shape = get_shape(env.observation_space.sample())
    act_shape = get_shape(env.action_space.sample())
    buffer = VPGBuffer(steps_per_epoch, obs_shape, act_shape,
                       gamma=gamma, lam=lam)

    # initialize actor
    rng, init_rng = random.split(rng)
    actor_input_spec = [
        (obs_shape, jnp.float32),
        (act_shape, jnp.int32),
    ]
    actor_def = actor_def.partial(action_space=env.action_space)
    _, initial_params = actor_def.init_by_shape(
        init_rng, actor_input_spec, act_rng=random.PRNGKey(0))
    actor = nn.Model(actor_def, initial_params)

    # initialize critic
    rng, init_rng = random.split(rng)
    critic_input_spec = [(obs_shape, jnp.float32)]
    _, initial_params = critic_def.init_by_shape(init_rng, critic_input_spec)
    critic = nn.Model(critic_def, initial_params)

    # create optimizers for actor and critic
    actor_optimizer = optim.Adam(learning_rate=actor_lr).create(actor)
    actor_optimizer = jax.device_put(actor_optimizer)
    critic_optimizer = optim.Adam(learning_rate=critic_lr).create(critic)
    critic_optimizer = jax.device_put(critic_optimizer)

    # define loss functions
    @jax.vmap
    def actor_loss(logp, adv):
        return -(logp * adv).mean()


    @jax.vmap
    def critic_loss(val, ret):
        return jnp.square(val - ret).mean()


    # define training steps for the actor and the critic
    @jax.jit
    def actor_train_step(rng, optimizer, obs, act, adv):
        def loss_fn(actor):
            pi, logp, logp_pi = actor(obs, act, rng)
            loss = jnp.mean(actor_loss(logp, adv))
            return loss
        loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, loss


    @jax.jit
    def critic_train_step(optimizer, obs, ret):
        def loss_fn(critic):
            val = critic(obs)
            loss = jnp.mean(critic_loss(val, ret))
            return loss
        loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, loss


    # define the function that consumes the batch to train
    def update(rng, actor_optimizer, critic_optimizer):
        obs, act, adv, ret, old_logp = buffer.get_batch()

        def eval(rng):
            # compute actor loss
            actor = actor_optimizer.target
            _, logp, _ = actor(obs, act, rng)
            actor_loss = jnp.mean(actor_loss(logp, adv))

            # compute critic loss
            critic = critic_optimizer.target
            val = critic(obs)
            critic_loss = jnp.mean(critic_loss(val, ret))

            return actor_loss, critic_loss, logp

        # evaluate before training
        rng, eval_rng = random.split(rng)
        old_actor_loss, old_critic_loss, _ = eval(eval_rng)

        # run policy gradient step
        rng, step_rng = random.split()
        actor_optimizer, actor_loss = actor_train_step(step_rng, actor_optimizer, obs, act, adv)

        # run value gradient steps
        for _ in range(train_value_iters):
            critic_optimizer, critic_loss = critic_train_step(critic_optimizer, obs, ret)

        # evaluate after training
        actor_loss, critic_loss, logp = eval(rng)

        # get metrics of the update
        approx_kl = (old_logp - logp).sum().item()
        approx_ent = (-logp).mean().item()
        metrics = {
            'actor_loss': old_actor_loss,
            'critic_loss': old_critic_loss,
            'kl': approx_kl,
            'entropy': approx_ent,
            'delta_actor_loss': (actor_loss - old_actor_loss),
            'delta_critic_loss': (critic_loss - old_critic_loss)
        }

        # log metrics
        # TODO: add tensorboard support
        logger.logkvs(metrics)

        return actor_optimizer, critic_optimizer


    start_time = time.time()
    obs, act = env.reset(), action_space.sample()
    ep_len, ep_ret = 0, 0
    actor_loss, critic_loss, logp = 0., 0., 0.
    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            rng, act_rng = random.split(rng)

            # get next action, its logprob and the state value
            act, _, logp_pi = actor(obs, act, act_rng)
            val = critic(obs).item()

            # take the action and store the step on the buffer
            new_obs, rew, done, _ = env.step(act.item())
            epoch_ended = buffer.store_timestep(obs, act, rew, val, logp_pi)

            # update obs
            obs = new_obs

            # end trajectory if necessary
            timeout = (ep_len == max_ep_len)
            terminal = (done or timeout)
            if terminal or epoch_ended:
                if not terminal:
                    print("Warning: trajectory cut of by epoch {:} at {:} steps.".format(epoch, t))
                # bootstrap last value if not at terminal state
                last_val = 0 if done else critic(obs).item()
                buffer.end_trajectory(last_val)
                if terminal:
                    logger.logkvs({
                        'ep_ret': ep_ret,
                        'ep_len': ep_len,
                    })
                obs, ep_ret, ep_len = env.reset(), 0, 0

        # save agent
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            agent_dict = {'actor': actor, 'critic': critic}
            save_to_zip(save_dir, agent_dict)

        # run VPG update
        rng, act_rng = random.split(rng)
        update(act_rng)

        # log epoch metrics
        logger.logkv('epoch', epoch)
        logger.logkv('time', time.time() - start_time)
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
                        help='Lmbda parameter of Generalized Advantage Estimation (GAE).')
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='Seed for the random number generator.')
    parser.add_argument('--steps', type=int, default=4000,
                        help='Number of training steps per epoch.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs.')
    parser.add_argument('--exp-name', type=str, default='vpg',
                        help='Experiment name for saving.')
    args = parser.parse_args()

    # Path where all experiment data and saves will be dumped
    save_dir = os.path.join('./experiments', args.exp_name)

    # Create agent def
    action_space = gym.make(args.env).action_space
    if isinstance(action_space, Box):
        actor_def = MLPGaussianActor.partial(hidden_sizes=(args.hidden_size,) * args.layers,
                                             activation_fn=nn.tanh, output_fn=None)
    elif isinstance(action_space, Discrete):
        actor_def = MLPCategoricalActor.partial(hidden_sizes=(args.hidden_size,) * args.layers,
                                                activation_fn=nn.tanh, output_fn=None)

    # Create critic def
    critic_def = MLPCritic.partial(hidden_sizes=(args.hidden_size,) * args.layers, activation_fn=nn.tanh)

    # Run experiment
    key = random.PRNGKey(args.seed)
    learn(key, lambda: gym.make(args.env), actor_def, critic_def,
          steps_per_epoch=args.steps, epochs=args.epochs, gamma=args.gamma,
          lam=args.lam, save_dir=save_dir)
