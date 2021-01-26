from rebar.learners.adp import ADP
from rebar.learners.qlearner import QLearner
import gym
import numpy as np
from gym import ObservationWrapper, ActionWrapper
from gym.spaces import Discrete
import pybulletgym

mins = np.array([-4.8, -4, -0.418, -4, -4])
maxes = -mins

env = gym.make('CartPole-v1')
env.observation_space.low=mins
env.observation_space.high=maxes

agt = ADP(
	env.action_space,
	env.observation_space,
	bins=20,
	initial_temp=2000,
	gamma=0.99
)

s = env.reset()
s_d = agt._convert_to_discrete(s)
sample = agt.sample_state(s_d, 10)

left_edge = np.array(s_d) - 1
right_edge = np.array(s_d)

print('===Bounds===')
print(agt.bounds)

print('===Discrete State Info===')
print(np.array(s_d))
print('===Edges===')
print(left_edge)
print(right_edge)

print('===State===')
print(s)
print('===Drawn States===')
print(sample)
