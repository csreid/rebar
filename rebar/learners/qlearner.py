import time
import copy

import numpy as np
from numpy import e
import torch
from torch.nn import Sequential, Linear, LeakyReLU, MSELoss, Conv2d, Flatten
from torch.optim import Adam, RMSprop
from torch.nn.utils import clip_grad_value_

from memory import Memory
from learner import Learner

np.seterr(all='raise')

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
		do_target=True,
		memory_len=10000,
		target_lag=None,
		exploration_steps=10000
	):
		self.n_actions = action_space.n
		self.action_space = action_space
		self._memory = Memory(memory_len, observation_space.shape)
		self.Q = Q

		if target_lag is not None:
			self.target_Q = copy.deepcopy(self.Q)
			self._target = True
			self._target_lag = target_lag
		else:
			self._target = False

		self.gamma = gamma

		self.opt = opt(self.Q.parameters(), **opt_args)
		self._base_loss_fn = MSELoss()
		self._steps = 0

		self.eps = 1.
		self.decay = 0.01 ** (1/exploration_steps)

	def learn(self, n_samples=64):
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

			for idx, t in enumerate(targets):
				t[int(a_s[idx].byte())] = r_s[idx] + self.gamma * vhat_sp_s[idx]

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
		l = self.learn()
		self._steps += 1

		if self._target and (self._steps % self._target_lag) == 0:
			self.target_Q = copy.deepcopy(self.Q)

	def get_action_vals(self, s):
		return self.Q(s[None, :])

	def exploration_strategy(self, s):
		self.eps *= self.decay
		if self.eps < 0.01:
			self.eps = 0.01

		if np.random.random() > self.eps:
			ps = np.zeros(self.n_actions)
			best_action = torch.argmax(self.Q(s[None, :]))
			ps[best_action] = 1.

			return np.random.choice(np.arange(len(ps)), p=ps)

		return self.action_space.sample()

	def deterministic_strategy(self, s):
		eps = 0.05
		if np.random.random() > eps:
			ps = np.zeros(self.n_actions)
			best_action = torch.argmax(self.Q(s[None, :])).detach().numpy()
			ps[best_action] = 1.
		else:
			ps = np.full(self.n_actions, 1 / self.n_actions)

		return ps

class TargetQLearning(QLearning):
	def __init__(
		self,
		n_actions,
		opt=Adam,
		opt_args={},
		loss=MSELoss,
		gamma=0.99,
		target_lag=100,
		transitions_per_fit=1,
		memory_len=10000
	):
		super().__init__(
			n_actions=n_actions,
			opt=opt,
			opt_args=opt_args,
			loss=loss,
			gamma=gamma,
			memory_len=memory_len
		)

		self.target_Q = copy.deepcopy(self.Q)
		self.target_lag = target_lag
		self.transitions_per_fit=transitions_per_fit

	def _build_dataset(self, n):
		with torch.no_grad():
			s_s, a_s, r_s, sp_s, done_mask = self._memory.sample(n)

			vhat_sp_s = torch.max(self.target_Q(sp_s.float()), dim=1).values
			vhat_sp_s[done_mask] = 0

			targets = self.Q(s_s.float())

			for idx, t in enumerate(targets):
				t[int(a_s[idx].byte())] = r_s[idx] + self.gamma * vhat_sp_s[idx]

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

		self._steps += 1
		if (self._steps % self.transitions_per_fit) == 0:
			l = self.learn()

		if (self._steps % self.target_lag) == 0:
			self.target_Q = copy.deepcopy(self.Q)
