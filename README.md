# Rebar

Tools for experiments with reinforcement learning with Torch and Python, so that they can look like this:

```py
env = gym.make('CartPole-v1')

Q_network = Sequential(
	Linear(4, 32),
	ReLU()
	Linear(32, 2)
)

agt = QLearner(env.action_space, env.observation_space, Q)

s = env.reset()
done = False
for step in range(5000):
	a = agt.get_action(s)
	sp, r, done, _ = env.step(a)

	agt.handle_transition(s, a, r, sp, done)

	s = sp
	if done:
		done = False
		s = env.reset()
```
