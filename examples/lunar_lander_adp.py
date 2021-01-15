from rebar.learners.adp import ADP
import gym
import numpy as np
from gym import ObservationWrapper

mins = np.array([-1, -1, -1, -1, -1, -1, -1, -1])
maxes = -mins

env = gym.make('LunarLander-v2').env
eval_env = gym.make('LunarLander-v2')

agt = ADP(
	env.action_space,
	env.observation_space,
	bins=5,
	mins=mins,
	maxes=maxes,
	initial_temp=5000,
	gamma=0.99
)

s = env.reset()
for step in range(5000):
	a = agt.get_action(s)
	sp, r, done, _ = env.step(a)

	agt.handle_transition(s, a, r, sp, done)

	s = sp
	if done:
		s = env.reset()

	if (step % 100) == 0:
		print(f'{step}: {agt.evaluate(eval_env, 10)}')
		agt.play(eval_env)

agt.play(eval_env)
