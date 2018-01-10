import tensorflow as tf
import numpy as np


def unison_shuffled_copies(a, b):
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]

def dice_coe(output, target, loss_type='jaccard', axis=[1,2,3], smooth=1e-5):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    loss_type : string
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : list of integer
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
        If both output and target are empty, it makes sure dice is 1.
        If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``,
        then if smooth is very small, dice close to 0 (even the image values lower than the threshold),
        so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`_
    """
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    ## old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    ## new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    dice = tf.reduce_mean(dice)
    return dice

def dirac_initializer_2d(filter_height, filter_width, input_dim, output_dim):

	temp = np.zeros([filter_height, filter_width, input_dim, output_dim])

	if(output_dim >= input_dim):
		for i in range(int(output_dim/input_dim)):
			temp[int(filter_width/2),int(filter_height/2), :, i*input_dim:(i+1)*input_dim] = np.eye(input_dim, dtype=np.float32)

	return temp

def ncrelu(x, name="crelu"):

	with tf.variable_scope(name) as scope:
		return tf.concat([tf.nn.relu(x), -tf.nn.relu(-x)], 3)


def conv2d_dirac(inputconv, output_dim=64, filter_height=5, filter_width=5, stride_height=1, stride_width=1, stddev=0.02, padding="SAME", name="conv2d", do_norm=True, norm_type='batch_norm', do_relu=False, relufactor=0):
	
	with tf.variable_scope(name) as scope:

		conv = tf.contrib.layers.conv2d(inputconv, output_dim, [filter_width, filter_height], [stride_width, stride_height], padding, 
			activation_fn=None, weights_initializer=tf.constant_initializer(dirac_initializer_2d(filter_height, filter_width, input_dim, output_dim)),
			biases_initializer=tf.constant_initializer(0.0))
		
		if do_norm:
			if norm_type == 'instance_norm':
				conv = instance_norm(conv)
			elif norm_type == 'batch_norm':
				conv = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope="batch_norm")

		if do_relu:
			if(relufactor == 0):
				conv = tf.nn.relu(conv,"relu")
			else:
				conv = lrelu(conv, relufactor, "lrelu")

		return conv


def dirac_conv2d(inputconv, output_dim=64, filter_height=5, filter_width=5, stride_height=1, stride_width=1, stddev=0.02, padding="SAME", do_norm=True, norm_type='batch_norm', name="dirac_conv2d"):

	with tf.variable_scope(name) as scope:

		input_dim = inputconv.get_shape().as_list()[-1]

		weight = tf.get_variable("weight",[filter_height, filter_width, input_dim, output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
		bias = tf.get_variable("bias",[output_dim], dtype=np.float32, initializer=tf.constant_initializer(0.0))

		dirac_weight = tf.get_variable("dirac_weight",[filter_height, filter_width, input_dim, output_dim], initializer=tf.constant_initializer(dirac_initializer_2d(filter_height, filter_width, input_dim, output_dim)), trainable=False)

		alpha = tf.get_variable("alpha", 1, initializer=tf.constant_initializer(1.0))
		beta = tf.get_variable("beta", 1, initializer=tf.constant_initializer(1.0))

		output_conv = tf.nn.conv2d(inputconv, alpha*dirac_weight + beta*weight, [1, stride_height, stride_width, 1], padding=padding)

		if do_norm:
			if norm_type == 'instance_norm':
				output_conv = instance_norm(output_conv)
			elif norm_type == 'batch_norm':
				output_conv = tf.contrib.layers.batch_norm(output_conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope="batch_norm")

		return output_conv
