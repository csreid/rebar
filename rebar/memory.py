import torch
import numpy as np

class Memory:
	def __init__(self, max_len, shape):
		self.s_s = torch.zeros((max_len,) + shape, requires_grad=False)
		self.a_s = torch.zeros((max_len), requires_grad=False)
		self.sp_s = torch.zeros((max_len,) + shape, requires_grad=False)
		self.r_s = torch.zeros((max_len), requires_grad=False)
		self.done_mask = np.zeros(max_len, dtype=bool)
		self._counter = 0
		self.max_len = max_len

	def __len__(self):
		return min(self._counter, self.max_len)

	def append(self, t):
		i = self._counter
		s, a, r, sp, done = t
		idx = i % self.max_len

		self.s_s[idx] = s
		self.a_s[idx] = a
		self.r_s[idx] = r
		self.sp_s[idx] = sp
		self.done_mask[idx] = done

		self._counter += 1

	def sample(self, n):
		idx = np.random.randint(0, min(self._counter, self.max_len), size=n)

		return (
			self.s_s[idx],
			self.a_s[idx],
			self.r_s[idx],
			self.sp_s[idx],
			self.done_mask[idx]
		)

class _SumTree:
	write = 0
	def __init__(self, max_len):
		self.max_len = max_len
		self.tree = torch.zeros(2 * max_len - 1)
		self.data = np.zeros(max_len, dtype=object)
		self._counter = 0

	def _propagate(self, idx, change):
		parent = (idx - 1) // 2
		self.tree[parent] += change

		if parent != 0:
			self._propagate(parent, change)

	def _retrieve(self, idx, s):
		left = 2 * idx + 1
		right = left + 1

		if left >= len(self.tree):
			return idx

		if s <= self.tree[left]:
			return self._retrieve(left, s)

		return self._retrieve(right, s - self.tree[left])

	def total(self):
		return self.tree[0]

	def add(self, p, data):
		idx = (self.write % self.max_len) + self.max_len - 1

		data_idx = self.write % self.max_len
		self.data[data_idx] = data
		self.update(idx, p)

		self.write += 1

		self._counter = min(self._counter+1, self.max_len)

	def update(self, idx, p):
		change = p - self.tree[idx]
		self.tree[idx] = p

		self._propagate(idx, change)

	def __len__(self):
		return self._counter

	def get(self, s):
		idx = self._retrieve(0, s)

		data_idx = idx - self.max_len + 1

		return (idx, self.tree[idx], self.data[data_idx])

class PrioritizedMemory:
	e = 0.01
	a = 0.6
	beta = 0.4
	beta_inc = .001

	def __init__(self, max_len):
		self.tree = _SumTree(max_len)
		self.max_len = max_len

	def _get_priority(self, err):
		return (np.abs(err) + self.e) ** self.a

	def append(self, t, err):
		p = self._get_priority(err)
		self.tree.add(p, t)

	def sample(self, n):
		batch = []
		idxs = []
		seg = self.tree.total() / n
		priorities = []

		self.beta = np.min([1., self.beta + self.beta_inc])

		for i in range(n):
			a = seg * i
			b = seg * (i + 1)

			s = np.random.uniform(a, b)
			idx, p, data = self.tree.get(s)
			priorities.append(p)
			batch.append(data)
			idxs.append(idx)

		priorities = np.array(priorities)
		#batch = np.array(batch)
		#idxs = np.array(idxs)

		p_s = priorities / self.tree.total()
		is_weight = np.power(len(self.tree) * p_s, -self.beta)
		is_weight /= is_weight.max()

		return batch, idxs, is_weight

	def update(self, idx, error):
		p = self._get_priority(error)
		self.tree.update(idx, p)
