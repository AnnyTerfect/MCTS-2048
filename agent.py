import numpy as np
import random
import copy
import time

class MCTSAgent:
	def __init__(self, env, gamma=0.99, c=100, iter_time=1):
		'''
		Q and N are saved by list in which
		Q[index] is a list where [0, 1, 2, 3] stands for [Q(s, 0), Q(s, 1), Q(s, 2), Q(s, 3)]
		And index are saved by dict I

		N is saved similarly

		I: state -> index where I[str(state)] is the index saved in Q and N
		'''
		self.env_backup = copy.deepcopy(env)
		self.Q = []
		self.N = []
		self.I = {}
		self.gamma = gamma
		self.c = c
		self.iter_time = iter_time

	def reset_env(self):
		'''
		reset the env by applying the backup
		'''
		self.env = copy.deepcopy(self.env_backup)

	def rollout(self, s, d):
		'''
		random rollout
		'''
		if d == 0:
			return 0
		a = random.randint(0, 3)
		obs, reward, done, info = self.env.step(a)
		if done:
			return reward
		return reward + self.gamma * self.rollout(obs, d - 1)

	def select_action(self, s, d):
		# record the start time
		t0 = time.time()

		self.reset_env()
		while True:
			self.simulate(s, d)
			# simulate only for 1 second
			if time.time() - t0 >= self.iter_time:
				break

		# search the list for action
		index = self.I[str(s)]
		Qs = [self.Q[index][i] for i in range(4)]
		action = np.argmax(Qs)
		return action

	def valid_action(self, s, a):
		s = np.array(s)

		# up or down
		if a == 0 or a == 1:
			return ((s != 0) * (s - np.vstack((np.zeros((1, 4), dtype='int'), s[: 3, :])) == 0)).sum().sum() > 0
		if a == 2 or a == 3:
			return ((s != 0) * (s - np.hstack((np.zeros((4, 1), dtype='int'), s[:, : 3])) == 0)).sum().sum() > 0

	def simulate(self, s, d):
		if d == 0:
			return 0

		# use str to save the historical node
		ss = str(s)

		# unseen state
		if ss not in self.I:
			self.I[ss] = len(self.N)
			self.Q.append([0, 0, 0, 0])
			self.N.append([0, 0, 0, 0])
			q = self.rollout(s, d)
			self.reset_env()
			return q
		
		# select a UCB best action
		index = self.I[ss]
		Ns = np.array(self.N[index])
		Qs = np.array(self.Q[index])
		cost = self.c * np.sqrt(np.log(sum(Ns) + 1) / (Ns + 1e-9))
		for i in range(4):
			Qs[i] += 0 if self.valid_action(s, i) else -100000
		a = np.argmax(Qs + cost)

		# do the action
		obs, reward, done, info = self.env.step(a)
		if done:
			self.reset_env()
			return reward

		# update
		q = reward + self.gamma * self.simulate(obs, d - 1)
		self.N[index][a] += 1
		self.Q[index][a] += (q - self.Q[index][a]) / self.N[index][a]

		# reset the env for next time
		self.reset_env()
		return q
