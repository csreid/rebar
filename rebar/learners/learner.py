import time
import numpy as np

class Learner:
	def __init__(self):
		self.name = None
		self._steps = 0

	def get_action(self, s, explore=True):
		if explore:
			return self.exploration_policy(s)

		return self.exploitation_policy(s)

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
		return evl

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

	def exploration_policy(self, s):
		raise NotImplementedError

	def exploitation_policy(self, s):
		raise NotImplementedError
