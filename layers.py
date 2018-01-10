# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Aug 19, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
from basic_layers import *
from dirac_layers import *
import tensorflow as tf

def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def weight_variable_devonc(shape, stddev=0.1):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W,keep_prob_):
    conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    return tf.nn.dropout(conv_2d, keep_prob_)

def deconv2d(x, W,stride):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='VALID')

def max_pool(x,n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')

def crop_and_concat(x1,x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)   

def pixel_wise_softmax(output_map):
    exponential_map = tf.exp(output_map)
    evidence = tf.add(exponential_map,tf.reverse(exponential_map,[False,False,False,True]))
    return tf.div(exponential_map,evidence, name="pixel_wise_softmax")

def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    return tf.div(exponential_map,tensor_sum_exp)



def cross_entropy(y_,output_map):
    return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map,1e-10,1.0)), name="cross_entropy")
#     return tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(output_map), reduction_indices=[1]))

def simple_block(inputconv, output_dim=64, filter_height=5, filter_width=5, stride_height=1, stride_width=1, stddev=0.02, padding="SAME", maxpool=False, upsample=False, do_norm=True, norm_type='batch_norm', name="dirac_conv2d"):
    # batchnorm -> relu -> (maxpool) -> diracconv(+batchnorm)-> dropout
    shortskip =inputconv
    conv = tf.contrib.layer.batch_norm(inputconv, decay=0.9, updates_collections=None, epsilon=1e-5, scope="batch_norm")
    conv = tf.nn.relu(conv, "relu")
    if (maxpool == True):
        conv = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        conv = dirac_conv2d(conv, output_dim, filter_height, filter_width, stride_height, stride_width, stddev, padding, do_norm, norm_type, name)
        conv = ncrelu(conv)
        conv = tf.nn.dropout(conv, self.dropout)
        shortskip = tf.nn.avg_pool(shortskip, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        conv = conv + shortskip
        return conv
    elif (upsample == True):
        conv = dirac_conv2d(conv, output_dim, filter_height, filter_width, stride_height, stride_width, stddev, padding, do_norm, norm_type, name)
        conv = ncrelu(conv)
        conv = tf.nn.dropout(conv, self.dropout)
        '''
        shortskip =
        '''
        conv = conv + shortskip
        return conv
    else:
        conv = dirac_conv2d(conv, output_dim, filter_height, filter_width, stride_height, stride_width, stddev, padding, do_norm, norm_type, name)
        conv = ncrelu(conv)
        conv = tf.nn.dropout(conv, self.dropout)
        conv = conv + shortskip
        return conv

def bottleneck(inputconv, output_dim=64, filter_height=5, filter_width=5, stride_height=1, stride_width=1, stddev=0.02, padding="SAME", maxpool=False, upsample=False, do_norm=True, norm_type='batch_norm'):
    # batchnorm -> relu -> (maxpool) -> diracconv(+batchnorm)-> dropout
    shortskip =inputconv
    conv = tf.contrib.layer.batch_norm(inputconv, decay=0.9, updates_collections=None, epsilon=1e-5, scope="batch_norm")
    conv = tf.nn.relu(conv, "relu")

    if (maxpool == True):
        conv = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        conv = dirac_conv2d(conv, output_dim, filter_height, filter_width, stride_height, stride_width, stddev, padding, do_norm, norm_type, name="dirac_conv2d")
        conv = ncrelu(conv)
        conv = tf.nn.dropout(conv, dropout)
        shortskip = tf.nn.avg_pool(shortskip, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        conv = conv + shortskip
        return conv
    elif (upsample == True):
        conv = dirac_conv2d(conv, output_dim, filter_height, filter_width, stride_height, stride_width, stddev, padding, do_norm, norm_type, name="dirac_conv2d")
        conv = ncrelu(conv)
        conv = tf.nn.dropout(conv, dropout)
        '''
        shortskip =
        '''
        conv = conv + shortskip
        return conv
    else:
        conv = dirac_conv2d(conv, output_dim, filter_height, filter_width, stride_height, stride_width, stddev, padding, do_norm, norm_type, name="dirac_conv2d")
        conv = ncrelu(conv)
        conv = tf.nn.dropout(conv, dropout)
        conv = conv + shortskip
        return conv

def residual_block(inputconv, num_iter=3, output_dim=64, filter_height=5, filter_width=5, stride_height=1, stride_width=1, stddev=0.02, padding="SAME", do_norm=True, norm_type='batch_norm', dropout=0.5, name="residualblock"):

    for i in range(num_iter):
        if i == 0:
            conv = dirac_conv2d(inputconv, output_dim, filter_height, filter_width, stride_height, stride_width, stddev, padding,do_norm, norm_type, name=name + "_" + str(i) +"_1")
        else:
            conv = dirac_conv2d(conv, output_dim, filter_height, filter_width, stride_height, stride_width, stddev, padding,do_norm, norm_type, name=name + "_" + str(i) +"_2")
        conv = tf.contrib.layers.conv2d(inputconv, output_dim/2, [1,1], [1,1], padding, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=stddev),biases_initializer=tf.constant_initializer(0.0))
        conv = ncrelu(conv)
        conv = tf.nn.dropout(conv, dropout)
        print(tf.shape(conv))

    return conv






































