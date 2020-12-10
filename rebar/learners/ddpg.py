import copy

from ..memory import Memory
from .learner import Learner
from ..utils.ou_process import OrnsteinUhlenbeckProcess
import torch
from torch.nn import MSELoss
from torch.optim import Adam

class DDPGLearner(Learner):
	def __init__(
		self,
		action_space,
		observation_space,
		actor_critic,
		opt=Adam,
		opt_args={},
		loss=MSELoss,
		gamma=0.99,
		memory_len=10000,
		tau=0.001,
		exploration_steps=10000
	):
		super().__init__()

		self.action_space = action_space
		self._memory = Memory(memory_len, observation_space.shape)
		self.ac_network = actor_critic
		self.target_ac_network = copy.deepcopy(actor_critic)

		self.tau = tau
		self.gamma = gamma
		self.actor_opt = opt(self.ac_network.mu_parameters())
		self.critic_opt = opt(self.ac_network.q_parameters())

		self.epsilon = 1.
		self.decay = 1. / exploration_steps
		self._base_loss_fn = loss()
		self._rp = OrnsteinUhlenbeckProcess(
			size=action_space.shape,
			theta=0.15,
			mu=0,
			sigma=0.2
		)

	def _build_dataset(self, n):
		with torch.no_grad():
			s_s, a_s, r_s, sp_s, done_s = self._memory.sample(n)
			vhat_sp_s = self.ac_network(sp_s).reshape(-1)

			q_target = r_s + 0.99 * (1 - torch.tensor(done_s).float()) * vhat_sp_s
			y = q_target

		return s_s, a_s, y

	def _soft_update(self):
		parameters = zip(
			self.target_ac_network.q_parameters(),
			self.ac_network.q_parameters()
		)
		for t_param, s_param in parameters:
			t_param.data.copy_(
				t_param.data * (1. - self.tau) + s_param.data * self.tau
			)

		parameters = zip(
			self.target_ac_network.mu_parameters(),
			self.ac_network.mu_parameters()
		)
		for t_param, s_param in parameters:
			t_param.data.copy_(
				t_param.data * (1. - self.tau) + s_param.data * self.tau
			)

	def learn(self, n_samples=64):
		if len(self._memory) < n_samples:
			return

		s_s, a_s, y = self._build_dataset(n_samples)
		y_pred = self.ac_network.critic(s_s, a_s).reshape(-1)

		self.critic_opt.zero_grad()
		loss = self._base_loss_fn(y_pred, y)
		loss.backward()
		self.critic_opt.step()

		self.actor_opt.zero_grad()
		actions = self.ac_network.actor(s_s)
		loss = -self.ac_network.critic(s_s, actions)
		loss = loss.mean()
		self.actor_opt.step()

		self._soft_update()

	def handle_transition(self, s, a, r, sp, done):
		self._memory.append((s, a, r, sp, done))
		self.learn()
		self._steps += 1

	def exploration_policy(self, s):
		a = self.ac_network.actor(s)
		a += self.epsilon * self._rp.sample()

		self.epsilon -= self.decay

		if self.epsilon < 0:
			self.epsilon = 0

		return a

	def exploitation_policy(self, s):
		return self.ac_network.actor(s)
