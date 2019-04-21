#
# different methods (raw python, TF) for performing batch normalization
#

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

def batch_norm_forward_np(X, gamma=1.0, beta=0.0):
	mu = np.mean(X, axis=0)
	var = np.var(X, axis=0)

	X_norm = (X - mu) / np.sqrt(var + 1e-8)

	out = gamma * X_norm + beta

	return out


def batch_norm_conv2d_tf(X):
	num_channels = X.get_shape()[-1].value

	batch_mean, batch_var = tf.nn.moments(X, [0], name='moments')

	gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
						name='gamma', trainable=False)
	beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
						name='beta', trainable=False)

	normed = tf.nn.batch_normalization(X, batch_mean, batch_var, beta, gamma, 1e-3)

	return normed


def plot_images(batch, plot_dim=4):
	f, axarr = plt.subplots(plot_dim, plot_dim)

	r_idxs = np.arange(batch.shape[0])
	# np.random.shuffle(r_idxs)
	plot_imgs = batch[r_idxs[:plot_dim**2]]

	for i in range(plot_dim):
		for j in range(plot_dim):
			axarr[i, j].imshow(np.squeeze(plot_imgs[i * plot_dim + j]), cmap="Greys_r")
	plt.show()