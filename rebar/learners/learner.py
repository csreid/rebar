import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from scipy.special import softmax

class Learner:
	def get_action(self, s, explore=True):
		if explore:
			ps = self.exploration_strategy(s)
			return np.random.choice(np.arange(len(ps)), p=ps)

		ps = self.deterministic_strategy(s)
		return np.random.choice(np.arange(len(ps)), p=ps)

	def evaluate(self, env, n, starting_state=None, max_steps=None):
		vals = []
		for _ in range(n):
			done = False
			s = env.reset()

			if starting_state is not None:
				s = starting_state
				env.env.state = s

			total_r = 0
			steps = 0

			while not done:
				a = self.get_action(s, explore=False)
				s, r, done, _ = env.step(a)
				total_r += r
				steps += 1

				if max_steps is not None and steps > max_steps:
					break

			vals.append(total_r)

		evl = np.mean(np.array(vals))
		self._last_eval = evl
		return np.mean(np.array(vals))

	def play(self, env):
		done = False
		s = env.reset()
		total_r = 0

		while not done:
			a = self.get_action(s, explore=False)
			s, r, done, _ = env.step(a)
			total_r += r

			env.render()
			time.sleep(1./60.)

		env.close()

	def set_name(self, name):
		self.name = name

	def handle_transition(self, s, a, r, sp, done):
		raise NotImplementedError

	def get_action_vals(self, s):
		raise NotImplementedError

	def exploration_strategy(self, s):
		raise NotImplementedError

	def deterministic_strategy(self, s):
		raise NotImplementedError
