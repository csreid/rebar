import numpy as np
from numpy import e
from itertools import product

from .learner import Learner
from IPython import embed
from scipy.special import softmax

import gym

class ADP(Learner):
	def __init__(
		self,
		action_space,
		observation_space,
		bins,
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

		self.mins = observation_space.low
		self.maxes = observation_space.high
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
		s = self._convert_to_discrete(s)

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

	def sample_state(self, s, n):
		# Take in a discretized state and return `n` random states in that "bin"

		if not all(type(i) == int for i in s):
			s = self._convert_to_discrete(s)

		left_edge = np.array(s) - 1
		right_edge = np.array(s)

		gen = np.array([
			np.random.uniform(low=self.bounds[idx][l], high=self.bounds[idx][r], size=n)
			for idx, (l, r)
			in enumerate(zip(left_edge, right_edge))
		]).T

		return gen

	def simulate_step(self, s, a, n=1):
		if not all(type(i) == int for i in s):
			s = self._convert_to_discrete(s)

		sp_s = []
		r_s = []
		p_s = []
		for sp in self.F[s][a]:
			p_sp, r = self.p_sp(s, a, sp)
			p_s.append(p_sp)
			r_s.append(r)
			sp_s.append(self.continuize_state(sp))

		ns = np.random.choice(len(sp_s), p=np.array(p_s), size=n)
		ns = np.array([
			sp_s[i]
			for i
			in ns
		])

		return ns

	def continuize_state(self, s):
		# Take a discretized state and return the middle continuous state value
		if not all(type(i) == int for i in s):
			s = self._convert_to_discrete(s)

		left_edge = np.array(s) - 1
		right_edge = np.array(s)

		low = np.array([self.bounds[idx][l] for idx, l in enumerate(left_edge)])
		high = np.array([self.bounds[idx][h] for idx, h in enumerate(right_edge)])

		return (low + high) / 2

	def action_probabilities(self, s, temp=None):
		if temp is None:
			temp = self._temp

		s = self._convert_to_discrete(s)

		qs = self.get_action_vals(s)
		ps = softmax(qs / temp)

		return ps

	def exploration_policy(self, s):
		s = self._convert_to_discrete(s)

		ps = self.action_probabilities(s)

		return np.random.choice(np.arange(len(ps)), p=ps)

	def exploitation_policy(self, s):
		try:
			s = self._convert_to_discrete(s)
			ps = self.action_probabilities(s, temp=1.)

			return np.random.choice(np.arange(len(ps)), p=ps)
		except Exception as exc:
			print(self.get_action_vals(s))
			raise exc
