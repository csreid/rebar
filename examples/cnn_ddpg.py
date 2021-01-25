from rebar.learners.ddpg import DDPGLearner
import gym
import torch
import math
import numpy as np
from torch import tanh
from torch.nn import Sequential, Linear, LeakyReLU, Module, Conv2d
from torch.nn.functional import leaky_relu
from gym import ObservationWrapper, ActionWrapper
from itertools import chain

class AW(ActionWrapper):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.action_space.shape = (2,)

	def action(self, a):
		real_action = np.array([a[0], 0, 0])

		if a[1] > 0:
			real_action[1] = a[1]
		elif a[1] < 0:
			real_action[2] = -a[1]

		return real_action

class TorchWrapper(ObservationWrapper, ActionWrapper):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.observation_space.shape = (3, 96, 96)
	def observation(self, obs):
		obs = np.copy(obs)
		return torch.tensor(
			np.moveaxis(obs, -1, 0)
		).float()

env = AW(TorchWrapper(gym.make('CarRacing-v0')))
eval_env = AW(TorchWrapper(gym.make('CarRacing-v0')))

print(env.observation_space)

def compute_out(inp, pad, stride, kern, dil):
	out = math.floor(
		((inp + 2 * pad - dil * (kern - 1) - 1) / stride) + 1
	)

	return out

def compute_output_shape(layer, input_shape):
	h_in, w_in = input_shape
	h_pad, w_pad = layer.padding
	h_stride, w_stride = layer.stride
	h_kern, w_kern = layer.kernel_size
	h_dil, w_dil = layer.dilation

	return (
		compute_out(h_in, h_pad, h_stride, h_kern, h_dil),
		compute_out(w_in, w_pad, w_stride, w_kern, w_dil)
	)

class ActorCritic(Module):
	def __init__(self, n_actions):
		super().__init__()

		self.shared_input = Conv2d(
			in_channels=3,
			out_channels=8,
			kernel_size=8,
			stride=4
		)
		out_shape = compute_output_shape(self.shared_input, (96, 96))

		self.shared_h1 = Conv2d(
			in_channels=8,
			out_channels=16,
			kernel_size=4,
			stride=2
		)
		out_shape = compute_output_shape(self.shared_h1, out_shape)

		self.shared_h2 = Conv2d(
			in_channels=16,
			out_channels=16,
			kernel_size=2,
			stride=1
		)
		out_shape = compute_output_shape(self.shared_h2, out_shape)

		self.shared_h3 = Linear(
			out_shape[0] * out_shape[1] * 16,
			32
		)

		self.mu_input = Linear(32, 64)
		self.mu_h1 = Linear(64, 64)
		self._mu = Linear(64, n_actions)

		self.q_input = Linear(32, 64)
		self.q_h1 = Linear(64 + n_actions, 64)
		self._Q = Linear(64, 1)

	def q_parameters(self):
		return chain(
			self.shared_input.parameters(),
			self.shared_h1.parameters(),
			self.shared_h2.parameters(),
			self.shared_h3.parameters(),
			self.q_input.parameters(),
			self.q_h1.parameters(),
			self._Q.parameters()
		)

	def mu_parameters(self):
		return chain(
			self.shared_input.parameters(),
			self.shared_h1.parameters(),
			self.shared_h2.parameters(),
			self.shared_h3.parameters(),
			self.mu_input.parameters(),
			self.mu_h1.parameters(),
			self._mu.parameters()
		)

	def reshape(self, s_s):
		if len(s_s.shape) == 3:
			s_s = s_s.reshape(1, 3, 96, 96)

		return s_s

	def shared(self, s_s):
		s_s = self.reshape(s_s)

		sh = self.shared_input(s_s)
		sh = leaky_relu(sh)
		sh = self.shared_h1(sh)
		sh = leaky_relu(sh)
		sh = self.shared_h2(sh)
		sh = leaky_relu(sh)
		sh = torch.flatten(sh, start_dim=1)
		sh = self.shared_h3(sh)
		sh = torch.sigmoid(sh)

		return sh

	def critic(self, s_s, a_s, do_shared=True):
		if len(a_s.shape) == 1:
			a_s = a_s.reshape(1, -1)

		if do_shared:
			features = self.shared(s_s)
		else:
			features = s_s

		q = self.q_input(features)
		q = leaky_relu(q)
		q = torch.cat((q, a_s), dim=1)
		q = self.q_h1(q)
		q = leaky_relu(q)
		q = self._Q(q)

		return q

	def actor(self, s_s, do_shared=True):
		if do_shared:
			features = self.shared(s_s)
		else:
			features = s_s

		mu = self.mu_input(features)
		mu = leaky_relu(mu)
		mu = self.mu_h1(mu)
		mu = leaky_relu(mu)
		mu = self._mu(mu)

		mu = tanh(mu)

		return mu

	def forward(self, s):
		s = self.reshape(s)

		sh = self.shared(s)

		mu = self.actor(sh, do_shared=False)
		q = self.critic(sh, mu, do_shared=False)

		return q

ac = ActorCritic(env.action_space.shape[0])
agt = DDPGLearner(
	env.action_space,
	env.observation_space,
	ac,
	gamma=0.99,
	exploration_steps=50000
)

s = env.reset()
for step in range(500000):
	a = agt.get_action(s)
	print(a)
	sp, r, done, _ = env.step(a)
	env.render()

	agt.handle_transition(s, torch.tensor(a), r, sp, done)

	s = sp
	if done:
		s = env.reset()

	if ((step+1) % 5000) == 0:
		print(f'{step}: {agt.evaluate(eval_env, 1)}')
	if ((step+1) % 500) == 0:
		agt.play(eval_env)

agt.play(eval_env)
