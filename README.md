# Rebar

Tools for experiments with reinforcement learning with Torch and Python, so that they can look like this:

```py
env = gym.make('CartPole-v1')

Q_network = Sequential(
	Linear(4, 32),
	ReLU()
	Linear(32, 2)
)

agt_cfg = {
	"action_space": env.action_space,
	"observation_space": env.observation_space,
	"Q": Q
}

exp = Experiment(
	env,
	QLearner,
	agt_cfg,
	eval_freq=100,
	steps_per_trial=5000,
	n_trials=10
)

results = exp.run()
```
