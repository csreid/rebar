from rebar.learners.qlearner import QLearner
import gym
import torch
import time
from torch import tanh
from torch.nn import Sequential, Linear, LeakyReLU, Module
from torch.nn.functional import leaky_relu
from gym import ObservationWrapper, ActionWrapper
import pybulletgym
from itertools import chain

class TorchWrapper(ObservationWrapper):
	def observation(self, obs):
		return torch.tensor(obs).float()

env_name='CartPole-v1'
env = TorchWrapper(gym.make(env_name).env)
eval_env = TorchWrapper(gym.make(env_name))

agt = QLearner(
	env.action_space,
	env.observation_space,
	Q='simple',
	gamma=0.99,
	exploration_steps=5000
)

s = env.reset()
for step in range(50000):
	a = agt.get_action(s)
	sp, r, done, _ = env.step(a)

	agt.handle_transition(s, torch.tensor(a), r, sp, done)

	s = sp
	if done:
		s = env.reset()

	if (step % 1000) == 0:
		print(f'{step}: {agt.evaluate(eval_env, 10)}')
		agt.play(eval_env)

agt.play(eval_env)
