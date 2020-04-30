from functools import partial

import jax
import jax.random as random


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

    def loss(self, state, batch):
        params = self.get_params(state)
        return self._loss(params, batch)

    @partial(jax.jit, static_argnums=0)
    def init_state(self, rng, obs):
        params = self._init_params(rng, obs)
        return self._init(params)

    @partial(jax.jit, static_argnums=0)
    def update(self, i, state, batch):
        params = self.get_params(state)
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
            obs, act = batch.observations, batch.actions
            logp = actor.logp(params, obs, act)
            return loss_fn(batch, logp)

        super(ActorLearner, self).__init__(opt, actor.init_params, loss)
        self.actor = actor

    @partial(jax.jit, static_argnums=0)
    def step(self, state, rng, obs):
        params = self.get_params(state)
        return self.actor.step(params, rng, obs)

    @partial(jax.jit, static_argnums=0)
    def logp(self, state, obs, act):
        params = self.get_params(state)
        return self.actor.logp(params, obs, act)


class ActionCriticLearner(Learner):
    def __init__(
        self,
        critic,
        opt,
        loss_fn,
    ):
        def loss(params, batch):
            obs = batch.observations
            actions_vals = critic.action_values(params, obs)
            return loss_fn(batch, actions_vals)

        super(ActionCriticLearner, self).__init__(opt, critic.init_params, loss)
        self.critic = critic

    @partial(jax.jit, static_argnums=0)
    def action_values(self, state, obs):
        params = self.get_params(state)
        return self.critic.action_values(params, obs)


class StateCriticLearner(Learner):
    def __init__(
        self,
        critic,
        opt,
        loss_fn,
    ):
        def loss(params, batch):
            obs = batch.observations
            val = critic.state_value(params, obs)
            return loss_fn(batch, val)

        super(StateCriticLearner, self).__init__(opt, critic.init_params, loss)
        self.critic = critic

    @partial(jax.jit, static_argnums=0)
    def state_value(self, state, obs):
        params = self.get_params(state)
        return self.critic.state_value(params, obs)
