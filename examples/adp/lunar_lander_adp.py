from rebar.learners.adp import ADP
from rebar.learners.qlearner import QLearner
import gym
import numpy as np
from gym import ObservationWrapper, ActionWrapper
from gym.spaces import Discrete
import pybulletgym


env_name = 'LunarLander-v2'
env = gym.make(env_name).env
mins = -np.ones(env.observation_space.shape[0])
maxes = -mins
env.observation_space.low = mins
env.observation_space.high = -mins

eval_env = gym.make(env_name)
eval_env.render()
sample_env = gym.make(env_name)

sample_env.reset()
for _ in range(10):
	sample_s, r, done, _ = sample_env.step(0)

agt = ADP(
	env.action_space,
	env.observation_space,
	bins=6,
	initial_temp=10000,
	gamma=0.99
)

sample_s_discrete = agt._convert_to_discrete(sample_s)

s = env.reset()
all_states = []
generated_states = []

for step in range(5000):
	a = int(agt.get_action(s))
	sp, r, done, _ = env.step(a)

	s_d = agt._convert_to_discrete(s)
	agt.handle_transition(s, a, r, sp, done)

	s = sp
	if done:
		s = env.reset()

	if (step % 100) == 0:
		sample_action_probs = agt.action_probabilities(sample_s)
		av_s = agt.get_action_vals(sample_s_discrete)
		sample_Q_s = agt.get_action_vals(sample_s_discrete)
		print('=====')
		print(f'{step}: {agt.evaluate(eval_env, 10)}')
		agt.play(eval_env)

agt.play(eval_env)

