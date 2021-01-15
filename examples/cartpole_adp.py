from rebar.learners.adp import ADP
import gym
import numpy as np

env = gym.make('CartPole-v1').env
eval_env = gym.make('CartPole-v1')

mins = np.array([-4.8, -2, -0.418, -4])
maxes = -mins

agt = ADP(
	env.action_space,
	env.observation_space,
	bins=7,
	mins=mins,
	maxes=maxes,
	initial_temp=5000,
	gamma=0.9
)

s = env.reset()
for step in range(2000):
	a = agt.get_action(s)
	sp, r, done, _ = env.step(a)

	agt.handle_transition(s, a, r, sp, done)

	s = sp
	if done:
		s = env.reset()

	if (step % 100) == 0:
		print(f'{step}: {agt.evaluate(eval_env, 10)}')

	env.render()

agt.play(eval_env)
