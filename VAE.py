# -*- coding: utf-8 -*-
"""
    @Author  : LiuZhian
    @Time    : 2019/10/5 0005 下午 10:38
    @Comment : 
"""

import torch
import torch.nn as  nn
import torch.nn.functional as F


class VAE(nn.Module):
	def __init__(self, input_dim=784, h_dim=400, z_dim=20):
		super(VAE, self).__init__()
		# encoder part
		self.fc1 = nn.Linear(input_dim, h_dim)
		self.fc2 = nn.Linear(h_dim, z_dim)  # mu
		self.fc3 = nn.Linear(h_dim, z_dim)  # log_sigma

		# decoder part
		self.fc4 = nn.Linear(z_dim, h_dim)
		self.fc5 = nn.Linear(h_dim, input_dim)

	def forward(self, x):
		mu, log_sigma = self.encode(x)
		sampled_z = self.reparameterzie(mu, log_sigma)
		res = self.decode(sampled_z)

		return res, mu, log_sigma

	def encode(self, x):
		"""
		encoding part.
		:param x: input image
		:return: mu and log_(sigma**2)
		"""
		h = F.relu(self.fc1(x))
		mu = self.fc2(h)
		log_sigma = self.fc3(h)  # estimate log(sigma**2) actually
		return mu, log_sigma

	def reparameterzie(self, mu, log_sigma):
		"""
		Given a standard gaussian distribution epsilon ~ N(0,1),
		we can sample the random variable z as per z = mu + sigma * epsilon
		:param mu:
		:param log_sigma:
		:return: sampled z
		"""
		std = torch.exp(log_sigma * 0.5)
		eps = torch.randn_like(std)
		return mu + std * eps

	def decode(self, x):
		"""
		Given a sampled z, decode it back to image
		:param x:
		:return:
		"""
		h = F.relu(self.fc4(x))
		res = F.sigmoid(self.fc5(h))
		return res
