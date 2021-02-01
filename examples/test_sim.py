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
	bins=5,
	initial_temp=2000,
	gamma=0.99
)

s = env.reset()
for _ in range(50000):
	a = agt.get_action(s)
	sp, r, done, _ = env.step(a)

	agt.handle_transition(s, a, r, sp, done)
	s = sp
	if done:
		done = False
		s = env.reset()

s = env.reset()
discrete_state = agt._convert_to_discrete(s)
taken_action = np.sum(agt.F[discrete_state][0][sp] for sp in agt.F[discrete_state][0])
print(s)
print(agt.continuize_state(s))
print(f'Seen the state {agt.visits[agt._convert_to_discrete(s)]} times; taken the action {taken_action}')
print(np.mean(agt.simulate_step(s, 0, n=50), axis=0))
s, r, done, _ = env.step(a)
print(s)
