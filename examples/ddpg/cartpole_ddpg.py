from rebar.learners.ddpg import DDPGLearner
import gym
import torch
import time
from torch import tanh
from torch.nn import Sequential, Linear, LeakyReLU, Module
from torch.nn.functional import leaky_relu
from gym import ObservationWrapper, ActionWrapper
import pybulletgym
from itertools import chain

class TorchWrapper(ObservationWrapper):
	def observation(self, obs):
		return torch.tensor(obs).float()

class ActorCritic(Module):
	def __init__(self, n_actions, n_obs):
		super().__init__()
		self.mu_input = Linear(n_obs, 64)
		self.mu_h1 = Linear(64, 64)
		self._mu = Linear(64, n_actions)

		self.q_input = Linear(n_obs, 64)
		self.q_h1 = Linear(64 + n_actions, 64)
		self._Q = Linear(64, 1)

	def q_parameters(self):
		return chain(
			self.q_input.parameters(),
			self.q_h1.parameters(),
			self._Q.parameters()
		)

	def mu_parameters(self):
		return chain(
			self.mu_input.parameters(),
			self.mu_h1.parameters(),
			self._mu.parameters()
		)

	def critic(self, s_s, a_s):
		if len(s_s.shape) == 1:
			s_s = s_s.reshape(1, -1)
		if len(a_s.shape) == 1:
			a_s = a_s.reshape(1, -1)

		q = self.q_input(s_s)
		q = leaky_relu(q)
		q = torch.cat((q, a_s), dim=1)
		q = self.q_h1(q)
		q = leaky_relu(q)
		q = self._Q(q)

		return q

	def actor(self, s_s):
		if (len(s_s.shape) == 1):
			s_s = s_s.reshape(1, -1)

		mu = self.mu_input(s_s)
		mu = leaky_relu(mu)
		mu = self.mu_h1(mu)
		mu = leaky_relu(mu)
		mu = self._mu(mu)

		mu = tanh(mu)

		return mu

	def forward(self, s):
		if (len(s.shape) == 1):
			s = s.reshape(1, -1)

		mu = self.actor(s)
		q = self.critic(s, mu)

		return q

env_name='InvertedPendulumPyBulletEnv-v0'
env = TorchWrapper(gym.make(env_name).env)
eval_env = TorchWrapper(gym.make(env_name))

env.render()

ac = ActorCritic(env.action_space.shape[0], env.observation_space.shape[0])
agt = DDPGLearner(
	env.action_space,
	env.observation_space,
	ac,
	gamma=0.99,
	exploration_steps=15000
)

s = env.reset()
for step in range(50000):
	a = agt.get_action(s)
	sp, r, done, _ = env.step(a)

	agt.handle_transition(s, torch.tensor(a), r, sp, done)

	s = sp
	if done:
		s = env.reset()

	if (step % 1000) == 0:
		print(f'{step}: {agt.evaluate(eval_env, 10)}')
		#agt.play(play_env)

agt.play(eval_env)

