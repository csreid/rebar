from rebar.learners.adp import ADP
from rebar.learners.qlearner import QLearner
import gym
import numpy as np
from gym import ObservationWrapper, ActionWrapper
from gym.spaces import Discrete
import pybulletgym

mins = np.array([-4.8, -4, -0.418, -4, -4])
maxes = -mins

env_name = 'CartPole-v1'
env = gym.make(env_name).env
env.observation_space.low = mins
env.observation_space.high = -mins

agt = ADP(
	env.action_space,
	env.observation_space,
	bins=9,
	initial_temp=2000,
	gamma=0.99
)

s = env.reset()

for step in range(5000):
	a = int(agt.get_action(s))
	sp, r, done, _ = env.step(a)

	s_d = agt._convert_to_discrete(s)
	agt.handle_transition(s, a, r, sp, done)

	s = sp
	if done:
		s = env.reset()

s = env.reset()
print(f's: {s}')
print(f'sims:')
agt.simulate(s, 0)
sp, r, done, _ = env.step(0)
print(f'Actual: {sp}')
