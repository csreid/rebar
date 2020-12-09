import copy

class Experiment:
	"""
	An experiment, which takes a `Learner` and performs a number of trials
	in the given environment.
	"""
	def init(
		self,
		env,
		agt,
		agt_cfg,
		eval_freq=None,
		steps_per_trial=1000,
		n_trials=1
	):
		""" Build an experiment """
		self.env = env
		self.eval_env = copy.deepcopy(env)
		self.agt = agt(**agt_cfg)
		self.results = []
		self.n_steps = steps_per_trial
		self.n_trials = n_trials
		self.eval_freq = eval_freq

	def _run_trial(self):
		done = False
		ep_r = 0
		s = self.env.reset()
		evals = None if self.eval_freq is None else []
		ep_vals = []

		for step in range(self.n_steps):
			a = self.agt.get_action(s)
			sp, r, done, _ = self.env.step(a)
			ep_r += r
			s = sp

			if done:
				s = self.env.reset()
				ep_vals.append(ep_r)
				ep_r = 0
				done = False

			if self.eval_freq is not None and ((step+1) % self.eval_freq) == 0:
				evl = self.agt.evaluate(self.eval_env, 1)
				evals.append(evl)

		if self.eval_freq is None:
			return ep_vals

		return (ep_vals, evals)

	def run(self):
		for _ in self.n_trials:
			self.results.append(self._run_trial())
