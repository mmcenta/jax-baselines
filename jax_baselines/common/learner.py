from functools import partial

import jax
import jax.random as random
import jax.numpy as jnp


class Learner:
    def __init__(
        self,
        opt,
        init_params,
        loss,
    ):
        self._init, self._update, self._get_params = opt
        self._init_params = init_params
        self._loss = loss

    def get_params(self, state):
        return self._get_params(state)

    @partial(jax.jit, static_argnums=0)
    def loss(self, state, batch):
        params = self._get_params(state)
        return self._loss(params, batch)

    @partial(jax.jit, static_argnums=0)
    def init_state(self, rng, obs):
        params = self._init_params(rng, obs)
        return self._init(params)

    @partial(jax.jit, static_argnums=0)
    def update(self, i, state, batch):
        params = self._get_params(state)
        g = jax.grad(self._loss)(params, batch)
        return self._update(i, g, state)


class ActorLearner(Learner):
    def __init__(
        self,
        actor,
        opt,
        loss_fn,
    ):
        def loss(params, batch):
            logp = actor.logp(params, batch['observations'], batch['actions'])
            return loss_fn(batch['advantages'], logp)

        super(ActorLearner, self).__init__(opt, actor.init_params, loss)
        self.actor = actor

    def step(self, state, rng, obs):
        params = self.get_params(state)
        return self.actor.step(params, rng, obs)

    def logp(self, state, obs, act):
        params = self.get_params(state)
        return self.actor.logp(params, obs, act)


def get_q_target_fn(q_value_fn, gamma=0.99):
    def get_targets(state, batch):
        next_q_values = q_value_fn(
            state['target_params'], batch['next_observations'])
        targets = (batch['rewards'] + gamma * next_q_values.max(axis=-1))
        targets = jnp.where(batch['is_terminal'], 0., targets)
        return targets

    return jax.jit(get_targets)

def get_double_q_target_fn(q_value_fn, get_params_fn, gamma=0.99):
    def get_targets(state, batch):
        batch_size = batch['observations'].shape[0]
        params = get_params_fn(state)

        # compute actions that maximize Q with online params
        next_q_values = q_value_fn(
            params['params'], batch['next_observations'])
        acts = next_q_values.argmax(axis=-1)

        # compute Q values with target params and the double q values
        target_next_q_values = q_value_fn(
            params['target_params'], batch['next_observations'])
        double_q_values = target_next_q_values[jnp.arange(batch_size), acts]

        # compute and return final targets
        targets = (batch['rewards'] + gamma * double_q_values)
        targets = jnp.where(batch['is_terminal'], 0., targets)
        return targets

    return jax.jit(get_targets)



class ActionCriticLearner(Learner):
    def __init__(
        self,
        critic,
        opt,
        loss_fn,
        gamma=0.99,
        polyak=False,
        tau=1e-2,
        target_train_steps=100,
        double_q=False,
        n_multistep=1,
    ):
        def loss(params, batch):
            q_vals = critic.action_values(params, batch['observations'])
            acts = batch['actions'].astype(jnp.int32)
            q_val = q_vals[jnp.arange(q_vals.shape[0]), acts]
            return loss_fn(batch['targets'], q_val)

        super(ActionCriticLearner, self).__init__(
            opt, critic.init_params, loss)

        if double_q:
            self._targets = get_double_q_target_fn(
                critic.action_values, self.get_params, gamma ** n_multistep)
        else:
            self._targets = get_q_target_fn(
                critic.action_values, gamma ** n_multistep)
        self.critic = critic
        self.gamma = gamma
        self.polyak = polyak
        self.tau = tau
        self.target_train_steps = target_train_steps

    def get_params(self, state):
        opt_state = state['opt_state']
        return {
            "params": self._get_params(opt_state),
            "target_params": state["target_params"],
        }

    def _targets(self, state, batch):
        next_q_values = self.critic.action_values(
            state['target_params'], batch['next_observations'])
        targets = (batch['rewards'] + self.gamma * next_q_values.max(axis=-1))
        targets = jnp.where(batch['is_terminal'], 0., targets)
        return targets

    def loss(self, state, batch):
        batch['targets'] = self._targets(state, batch)
        return super(ActionCriticLearner, self).loss(state['opt_state'], batch)

    def init_state(self, rng, obs):
        opt_state = super(ActionCriticLearner, self).init_state(rng, obs)
        target_params = self._get_params(opt_state)
        return {
            "opt_state": opt_state,
            "target_params": target_params,
        }

    def update(self, i, state, batch):
        # unpack state
        opt_state = state['opt_state']
        target_params = state['target_params']

        # add targets to batch
        batch['targets'] = self._targets(state, batch)

        # execute optimizer step and recover new parameters
        opt_state = super(ActionCriticLearner, self).update(
            i, opt_state, batch)
        params = self._get_params(opt_state)

        # update target parameters
        if self.polyak:
            target_params = jax.tree_multimap(
                lambda p1, p2: (1 - self.tau) * p1 + self.tau * p2,
                target_params, params)
        elif i % self.target_train_steps == 0:
            target_params = params

        return {
            "opt_state": opt_state,
            "target_params": target_params,
        }

    def action_values(self, state, obs, train=True):
        if train:
            params = self.get_params(state)['params']
        else:
            params = self.get_params(state)['target_params']
        return self.critic.action_values(params, obs)


class StateCriticLearner(Learner):
    def __init__(
        self,
        critic,
        opt,
        loss_fn,
    ):
        def loss(params, batch):
            val = critic.state_value(params, batch['observations'])
            return loss_fn(batch['returns'], val)

        super(StateCriticLearner, self).__init__(opt, critic.init_params, loss)
        self.critic = critic

    def state_value(self, state, obs):
        params = self.get_params(state)
        return self.critic.state_value(params, obs)
