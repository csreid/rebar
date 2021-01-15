import numpy as np
from numpy import e
from itertools import product

from .learner import Learner
from IPython import embed

import gym

class ADP(Learner):
	def __init__(
		self,
		action_space,
		observation_space,
		bins,
		mins,
		maxes,
		initial_temp=5000,
		gamma=0.99,
		delta=0.1
	):
		super().__init__()
		self.bins = bins
		self.n_actions = action_space.n
		self.n_obs = observation_space.shape[0]
		self.gamma = gamma
		self.delta = delta

		self.mins = mins
		self.maxes = maxes
		sp_shape = [bins+1 for _ in range(self.n_obs)]
		sp_shape[0] += 1
		sp_shape = tuple(sp_shape)
		self._temp = initial_temp

		self.F = {}
		self.R = {}

		self._terminal_state = tuple([i-1 for i in sp_shape])

		self.V = np.zeros(tuple(
			[bins+1 for _ in range(self.n_obs)]
		))
		self.visits = np.zeros(self.V.shape)

		self.bounds = np.array([
			np.linspace(self.mins[i], self.maxes[i], bins)
			for i in range(self.n_obs)
		])

		self._statemap = {}
		for s in product(*[range(bins+1) for _ in range(self.n_obs)]):
			for a in range(self.n_actions):
				self._statemap[s + (a,)] = []

	def p_sp(self, s, a, sp):
		expected_reward = np.mean(self.R[tuple(s)][a][tuple(sp)])

		total_s_a = 0
		for tmp in self.F[tuple(s)][a]:
			total_s_a += self.F[tuple(s)][a][tmp]

		p_sp = (self.F[tuple(s)][a][tuple(sp)]) / total_s_a

		return p_sp, expected_reward

	def handle_transition(self, s, a, r, sp, done):
		s = self._convert_to_discrete(s)
		sp = self._convert_to_discrete(sp)

		if done:
			sp = self._terminal_state

		self.visits[tuple(s)] += 1
		if tuple(s) not in self.F:
			self.F[tuple(s)] = {}
		if a not in self.F[tuple(s)]:
			self.F[tuple(s)][a] = {}
		if tuple(sp) not in self.F[tuple(s)][a]:
			self.F[tuple(s)][a][tuple(sp)] = 0

		self.F[tuple(s)][a][tuple(sp)] += 1

		if tuple(s) not in self.R:
			self.R[tuple(s)] = {}
		if a not in self.R[tuple(s)]:
			self.R[tuple(s)][a] = {}
		if tuple(sp) not in self.R[tuple(s)][a]:
			self.R[tuple(s)][a][tuple(sp)] = []

		self.R[tuple(s)][a][tuple(sp)].append(r)

		if tuple(s) + (a,) in self._statemap:
			self._statemap[tuple(s) + (a,)].append(sp)
		else:
			self._statemap[tuple(s) + (a,)] = [sp]

		self._temp = max(1, self._temp-1)
		passes = 0
		while self.do_pass() > self.delta:
			passes += 1

	def do_pass(self):
		delta = 0
		for s in self.F:
			v = self.V[s]
			self.V[s] = np.max(self.get_action_vals(s))
			delta = np.max([delta, np.abs(v - self.V[s])])

		return delta

	def get_action_vals(self, s):
		vals = []
		for a in range(self.n_actions):
			total = 0
			for sp in set(self._statemap[tuple(s) + (a,)]):
				p_sp, e_r = self.p_sp(s, a, sp)
				if sp == self._terminal_state:
					v_sp = 0
				else:
					v_sp = self.V[sp]

				total += p_sp * (e_r + self.gamma * v_sp)

			vals.append(total)

		return np.array(vals)

	def _convert_to_discrete(self, s):
		bounds = self.bounds
		if type(s) is tuple:
			return s

		new_obs = tuple(
			np.searchsorted(bounds[i], s[i])
			for i in range(self.n_obs)
		)

		return new_obs

	def exploration_policy(self, s):
		s = self._convert_to_discrete(s)

		qs = self.get_action_vals(s)
		ps = [(e ** (q / self._temp)) / np.sum(e ** (qs / self._temp)) for q in qs]
		return np.random.choice(np.arange(len(ps)), p=ps)

	def exploitation_policy(self, s):
		s = self._convert_to_discrete(s)
		qs = self.get_action_vals(s)
		temp=1.
		ps = [(e ** (q / temp)) / np.sum(e ** (qs / temp)) for q in qs]

		return np.random.choice(np.arange(len(ps)), p=ps)
