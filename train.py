# -*- coding: utf-8 -*-
"""
    @Author  : LiuZhian
    @Time    : 2019/10/5 0005 下午 10:24
    @Comment : 
"""
import argparse
import torch
import torch.optim as  optim
from torchvision.utils import save_image
from VAE import *
from utils import prepare_MNIST
import os
from convVAE import *
import shutil
import numpy as np

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

parser = argparse.ArgumentParser(description='Convolutional VAE MNIST Example')
parser.add_argument('--result_dir', type=str, default='./convVAEresult', metavar='DIR',
					help='output directory')
parser.add_argument('--save_dir', type=str, default='./ckpt', metavar='DIR',
					help='model saving directory')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
					help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
					help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: None')
parser.add_argument('--test_every', default=10, type=int, metavar='N',
					help='test after every epochs')
parser.add_argument('--num_worker', type=int, default=2, metavar='N',
					help='num_worker')

# model options
parser.add_argument('--lr', type=float, default=1e-3,
					help='learning rate')
parser.add_argument('--z_dim', type=int, default=20, metavar='N',
					help='latent vector size of encoder')
parser.add_argument('--input_dim', type=int, default=28 * 28, metavar='N',
					help='input dimension (28*28 for MNIST)')
parser.add_argument('--input_channel', type=int, default=1, metavar='N',
					help='input channel (1 for MNIST)')

args = parser.parse_args()

kwargs = {'num_workers': 2, 'pin_memory': True} if cuda else {}


def save_checkpoint(state, is_best, outdir):

	if not os.path.exists(outdir):
		os.makedirs(outdir)

	checkpoint_file = os.path.join(outdir, 'checkpoint.pth')
	best_file = os.path.join(outdir, 'model_best.pth')
	torch.save(state, checkpoint_file)
	if is_best:
		shutil.copyfile(checkpoint_file, best_file)


def loss_func(recon_x, inputs, mu, log_sigma):
	# Calculate the loss. Note that the loss includes two parts.
	# 1. the reconstruction loss.
	# We regard the MNIST as binary classification
	reconstruction_loss = F.binary_cross_entropy(recon_x, inputs, reduction='sum')

	# 2. KL-divergence
	# D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
	divergence = 0.5 * torch.sum(torch.exp(log_sigma) + torch.pow(mu, 2) - 1. - log_sigma)

	loss = reconstruction_loss + divergence

	return loss, reconstruction_loss, divergence


def train():
	myVAE = ConvVAE(input_channels=args.input_channel, z_dim=args.z_dim).to(device)
	# myVAE = VAE(INPUT_DIM, H_DIM, Z_DIM).to(device)
	optimizer = optim.Adam(myVAE.parameters(), lr=args.lr)

	start_epoch = 0
	best_test_loss = np.finfo('f').max

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print('=> loading checkpoint %s' % args.resume)
			checkpoint = torch.load(args.resume)
			start_epoch = checkpoint['epoch'] + 1
			best_test_loss = checkpoint['best_test_loss']
			myVAE.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print('=> loaded checkpoint %s' % args.resume)
		else:
			print('=> no checkpoint found at %s' % args.resume)

	if not os.path.exists(args.result_dir):
		os.makedirs(args.result_dir)

	trainloader, testloader, classes = prepare_MNIST(args.batch_size, args.num_worker)


	# training
	for epoch in range(start_epoch, args.epochs):

		for i, data in enumerate(trainloader):
			# get the inputs; data is a list of [inputs, labels]
			# Remember to deploy the input data on GPU
			inputs = data[0].to(device)

			# forward
			res, mu, log_sigma = myVAE(inputs)

			loss, recon_loss, KLD = loss_func(res, inputs, mu, log_sigma)
			# zero out the paramter gradients
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# print statistics every 100 batches
			if (i + 1) % 100 == 0:
				print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f} Total loss {:.4f}"
					  .format(epoch + 1, args.epochs, i + 1, len(trainloader), recon_loss.item(),
							  KLD.item(), loss.item()))

			if i == 0:
				# visualize reconstructed result at the beginning of each epoch
				x_concat = torch.cat([inputs.view(-1, 1, 28, 28), res.view(-1, 1, 28, 28)], dim=3)
				save_image(x_concat, ("./%s/reconstructed-%d.png" % (args.result_dir, epoch + 1)))

		# testing
		if (epoch + 1) % args.test_every == 0:
			test_avg_loss = 0.0
			with torch.no_grad():
				for idx, test_data in enumerate(testloader):
					# get the inputs; data is a list of [inputs, labels]
					test_inputs = test_data[0].to(device)
					# forward
					test_res, test_mu, test_log_sigma = myVAE(test_inputs)

					test_loss, test_recon_loss, test_KLD = loss_func(test_res, test_inputs, test_mu, test_log_sigma)

					test_avg_loss += test_loss

				test_avg_loss /= len(testloader.dataset)

				# we randomly sample some images' latent vectors from its distribution
				z = torch.randn(args.batch_size, args.z_dim).to(device)
				random_res = myVAE.decode(z).view(-1, 1, 28, 28)
				save_image(random_res, "./%s/random_sampled-%d.png" % (args.result_dir, epoch + 1))

				# save model
				is_best = test_avg_loss < best_test_loss
				best_test_loss = min(test_avg_loss, best_test_loss)
				save_checkpoint({
					'epoch': epoch,
					'best_test_loss': best_test_loss,
					'state_dict': myVAE.state_dict(),
					'optimizer': optimizer.state_dict(),
				}, is_best, args.save_dir)



if __name__ == '__main__':
	train()
