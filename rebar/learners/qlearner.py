import copy

import numpy as np
import torch
from torch.nn import  MSELoss, Module, Linear, LeakyReLU
from torch.nn.functional import leaky_relu
from torch.optim import Adam

from ..memory import Memory
from .learner import Learner

np.seterr(all='raise')

class SimpleFFNN(Module):
	def __init__(self, action_space, observation_space):
		super().__init__()
		self.n_actions = action_space.n
		self.obs_size = observation_space.shape[0]

		self.input = Linear(self.obs_size, 2 * self.obs_size)
		self.hidden = Linear(2 * self.obs_size, 2 * self.obs_size)
		self.output = Linear(2 * self.obs_size, self.n_actions)

	def forward(self, s):
		if (len(s.shape) == 1):
			s = s.reshape(1, -1)

		out = self.input(s)
		out = leaky_relu(out)
		out = self.hidden(out)
		out = leaky_relu(out)
		out = self.output(out)

		return out

class QLearner(Learner):
	def __init__(
		self,
		action_space,
		observation_space,
		Q,
		opt=Adam,
		opt_args={},
		loss=MSELoss,
		gamma=0.99,
		memory_len=10000,
		target_lag=None,
		exploration_steps=10000,
		initial_epsilon=1.0,
		final_epsilon=0.01
	):
		super().__init__()

		if Q == 'simple':
			Q = SimpleFFNN(action_space, observation_space)

		self.n_actions = action_space.n
		self.action_space = action_space
		self._memory = Memory(memory_len, observation_space.shape, (1,))
		self.Q = Q

		if target_lag is not None:
			self.target_Q = copy.deepcopy(self.Q)
			self._target = True
			self._target_lag = target_lag
		else:
			self._target = False

		self.gamma = gamma

		self.opt = opt(self.Q.parameters(), **opt_args)
		self._base_loss_fn = loss()
		self._steps = 0

		self.eps = initial_epsilon
		self.decay = final_epsilon ** (1/exploration_steps)

	def learn(self, n_samples=32):
		if len(self._memory) < n_samples:
			return 'n/a'

		X, y = self._build_dataset(n_samples)
		y_pred = self.Q(X)
		loss = self._base_loss_fn(y, y_pred)

		self.opt.zero_grad()
		loss.backward()
		self.opt.step()

		return loss.item()

	def _build_dataset(self, n):
		with torch.no_grad():
			s_s, a_s, r_s, sp_s, done_mask = self._memory.sample(n)

			if self._target:
				q = self.target_Q
			else:
				q = self.Q

			vhat_sp_s = torch.max(q(sp_s.float()), dim=1).values
			vhat_sp_s[done_mask] = 0

			targets = self.Q(s_s.float())

			for idx, target in enumerate(targets):
				target[int(a_s[idx].byte())] = r_s[idx] + self.gamma * vhat_sp_s[idx]

			X = s_s.float()
			y = targets
		return X, y

	def handle_transition(self, s, a, r, sp, done):
		self._memory.append((
			s,
			torch.from_numpy(np.array([a]))[0],
			r,
			sp,
			done
		))
		self.learn()
		self._steps += 1

		if self._target and (self._steps % self._target_lag) == 0:
			self.target_Q = copy.deepcopy(self.Q)

	def get_action_vals(self, s):
		return self.Q(s[None, :])

	def exploration_policy(self, s):
		self.eps *= self.decay
		if self.eps < 0.01:
			self.eps = 0.01

		if np.random.random() > self.eps:
			best_action = torch.argmax(self.Q(s[None, :])).detach().numpy()
			a = best_action

		else:
			a = self.action_space.sample()

		return a

	def exploitation_policy(self, s):
		eps = 0.05
		if np.random.random() > eps:
			best_action = torch.argmax(self.Q(s[None, :])).detach().numpy()
			a = best_action
		else:
			probabilities = np.full(self.n_actions, 1 / self.n_actions)
			a = np.random.choice(np.arange(len(probabilities)), p=probabilities)

		return a
