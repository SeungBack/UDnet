import tensorflow as tf
import numpy as np
import optparse
import os
import shutil
import time
import random
import sys
import pickle
import wget
import tarfile

from basic_layers import *
from dirac_layers import *
from layers import *
from isbi_utils import *

'''
hyperparameters

batch_norm decay : 
Decay for the moving average. Reasonable values for decay are close to 1.0, 
typically in the multiple-nines ran/ge: 0.999, 0.99, 0.9, etc. Lower decay value (recommend trying decay=0.9) 
if model experiences reasonably good training performance but poor validation and/or test performance. 
Try zero_debias_moving_mean=True for improved stability.

'''

class UDnet():

    def run_parser(self):
        self.parser = optparse.OptionParser()

        self.parser.add_option('--num_iter', type='int', default=1000, dest='num_iter')
        self.parser.add_option('--batch_size', type='int', default=100, dest='batch_size')
        self.parser.add_option('--img_width', type='int', default=512, dest='img_width')
        self.parser.add_option('--img_height', type='int', default=512, dest='img_height')
        self.parser.add_option('--img_depth', type='int', default=1, dest='img_depth')
        self.parser.add_option('--num_groups', type='int', default=3, dest='num_groups')
        self.parser.add_option('--num_blocks', type='int', default=4, dest='num_blocks')
        self.parser.add_option('--num_images', type='int', default=30, dest='num_images')
        self.parser.add_option('--num_test_images', type='int', default=100, dest='num_test_images')
        self.parser.add_option('--max_epoch', type='int', default=20, dest='max_epoch')
        self.parser.add_option('--n_samples', type='int', default=50000, dest='n_samples')
        self.parser.add_option('--test', action="store_true", default=False, dest="test")
        self.parser.add_option('--steps', type='int', default=10, dest='steps')
        self.parser.add_option('--enc_size', type='int', default=256, dest='enc_size')
        self.parser.add_option('--dec_size', type='int', default=256, dest='dec_size')
        self.parser.add_option('--model', type='string', default="draw_attn", dest='model_type')
        self.parser.add_option('--dataset', type='string', default="isbi", dest='dataset')
        self.parser.add_option('--dataset_folder', type='string', default="../../../datasets", dest='dataset_folder')
        self.parser.add_option('--dropout', type='float', default=0.5, dest='dropout')

    def __init__(self):
        self.run_parser()
        opt = self.parser.parse_args()[0]

        self.max_epoch = opt.max_epoch
        self.batch_size = opt.batch_size
        self.dataset = opt.dataset
        if (self.dataset == "isbi"):
            self.img_width = 512
            self.img_height = 512
            self.img_depth = 1
        else :
            self.img_width = opt.img_width
            self.img_height = opt.img_height
            self.img_depth = opt.img_depth

        self.img_size = self.img_width * self.img_height * self.img_depth
        self.num_groups = opt.num_groups
        self.num_blocks = opt.num_blocks
        self.num_images = opt.num_images
        self.num_test_images = opt.num_test_images
        self.dataset_folder = opt.dataset_folder
        self.model = "dirac"
        self.to_test = opt.test
        self.load_checkpoint = False
        self.do_setup = True
        self.dropout = opt.dropout

        self.tensorboard_dir = "./output/" + self.model + "/" + self.dataset + "/tensorboard"
        self.check_dir = "./output/" + self.model + "/" + self.dataset + "/checkpoints"
        self.images_dir = "./output/" + self.model + "/" + self.dataset + "/imgs"

    def load_isbi_dataset(self, mode='train'):

        if self.dataset == 'isbi':

            if (mode == 'train'):

                self.train_images, self.train_labels = isbi_get_data_montage('/home/seung/UDnet/ISBI/train-volume.tif', '/home/seung/UDnet/ISBI/train-labels.tif', nb_rows=6, nb_cols=5, rng=np.random)

            elif (mode == 'test'):

                self.test_images = np.reshape(self.test_images,[self.num_images, self.img_height, self.img_width, self.img_depth])
        else:
            print("Model not supported for this dataset")
            sys.exit()

    def model_setup(self):

        with tf.variable_scope("Model") as scope:

            self.input_imgs = tf.placeholder(tf.float32,
                                             [self.batch_size, self.img_height, self.img_width, self.img_depth])
            self.input_labels = tf.placeholder(tf.int32, [self.batch_size])

            if (self.dataset == 'cifar-10'):
                self.cifar_model_setup()
            elif (self.dataset == 'Imagenet'):
                self.inet_model_setup()
            elif (self.dataset == 'isbi'):
                self.isbi_model_setup()
            else:
                print("No such dataset exist. Exiting the program")
                sys.exit()

        self.model_vars = tf.trainable_variables()
        for var in self.model_vars: print(var.name, var.get_shape())

        self.do_setup = False

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
        elif (upsample == True)
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
            conv = tf.nn.dropout(conv, self.dropout)
            shortskip = tf.nn.avg_pool(shortskip, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            conv = conv + shortskip
            return conv
        elif (upsample == True)
            conv = dirac_conv2d(conv, output_dim, filter_height, filter_width, stride_height, stride_width, stddev, padding, do_norm, norm_type, name="dirac_conv2d")
            conv = ncrelu(conv)
            conv = tf.nn.dropout(conv, self.dropout)
            '''
            shortskip =
            '''
            conv = conv + shortskip
            return conv
        else:
            conv = dirac_conv2d(conv, output_dim, filter_height, filter_width, stride_height, stride_width, stddev, padding, do_norm, norm_type, name="dirac_conv2d")
            conv = ncrelu(conv)
            conv = tf.nn.dropout(conv, self.dropout)
            conv = conv + shortskip
            return conv


    def isbi_model_setup(self):

        self.input_imgs = tf.placeholder(tf.float32,[self.batch_size, self.img_height, self.img_width, self.img_depth]) # 512x512x1
        self.input_labels = tf.placeholder(tf.int32, [self.batch_size])

        input_pad = tf.pad(self.input_imgs, [[0, 0], [1, 1], [1, 1], [0, 0]])
        o_ctop = general_conv2d(input_pad, 32, 3, 3, 1, 1, do_norm=False, name="conv_top") # 512x512x32
        # (inputconv, output_dim, filter_height, filter_width, stride_height, stride_width, stddev, padding, name, do_norm=True, norm_type='batch_norm', do_relu=False, relufactor=0)
        #inputconv, output_dim=64, filter_height=5, filter_width=5, stride_height=1, stride_width=1, stddev=0.02, padding="SAME", maxpool=False, upsample=False, do_norm=True, norm_type='batch_norm'):

        o_simple1 = self.simple_block(o_ctop, 32, 3, 3, 1, 1, maxpool=True, name="simple_down") #256x256x32
        o_bottleneck1 = self.bottleneck() #128x128x128

        for group in range(0, self.num_groups):
            for block in range(0, self.num_blocks):
                o_loop = ncrelu(o_loop, name="crelu_" + str(group) + "_" + str(block))
                o_loop = dirac_conv2d(o_loop, outdim, 3, 3, 1, 1, name="conv_" + str(group) + "_" + str(block))

            if (group != self.num_groups - 1):
                o_loop = tf.nn.pool(o_loop, [2, 2], "MAX", "VALID", None, [2, 2], name="maxpool_" + str(group))

            outdim = outdim * 2

        temp_shape = o_loop.get_shape().as_list()
        o_avgpool = tf.nn.avg_pool(o_loop, [1, temp_shape[1], temp_shape[2], 1], [1, temp_shape[1], temp_shape[2], 1],
                                   "VALID", name="avgpool")
        temp_depth = o_avgpool.get_shape().as_list()[-1]
        self.final_output = linear1d(tf.reshape(o_avgpool, [self.batch_size, temp_depth]), temp_depth, 10)

    def loss_setup(self):

        eps = 1e-5
        prediction = pixel_wise_softmax_2(logits)
        intersection = tf.reduce_sum(prediction * self.y)
        union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(self.y)

        self.loss = -(2 * intersection / (union))
        self.loss = tf.reduce_mean(self.loss)
        optimizer = tf.train.RMSPropOptimizer(0.001, beta1=0.5)
        self.loss_optimizer = optimizer.minimize(self.loss)

        # Defining the summary ops
        self.cl_loss_summ = tf.summary.scalar("cl_loss", self.loss)

    # print(self.loss.shape)

    def train(self):

        self.model_setup()
        self.loss_setup()

        if self.dataset == 'isbi' :
            self.load_isbi_dataset('train')
            self.normalize_input(self.input_imgs)

        else:
            print('No such dataset exist')
            sys.exit()

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        if not os.path.exists(self.images_dir + "/train/"):
            os.makedirs(self.images_dir + "/train/")
        if not os.path.exists(self.check_dir):
            os.makedirs(self.check_dir)

        with tf.Session() as sess:

            sess.run(init)
            writer = tf.summary.FileWriter(self.tensorboard_dir)
            writer.add_graph(sess.graph)

            if self.load_checkpoint:
                chkpt_fname = tf.train.latest_checkpoint(self.check_dir)
                saver.restore(sess, chkpt_fname)

            for epoch in range(0, self.max_epoch):

                for itr in range(0, int(self.num_images / self.batch_size)):
                    imgs = self.train_images[itr * self.batch_size:(itr + 1) * (self.batch_size)]
                    labels = self.train_labels[itr * self.batch_size:(itr + 1) * (self.batch_size)]

                    _, summary_str, cl_loss_temp = sess.run([self.loss_optimizer, self.cl_loss_summ, self.loss],
                                                            feed_dict={self.input_imgs: imgs,
                                                                       self.input_labels: labels})

                    print("In the iteration " + str(itr) + " of epoch " + str(
                        epoch) + " with classification loss of " + str(cl_loss_temp))

                    writer.add_summary(summary_str, epoch * int(self.num_images / self.batch_size) + itr)

                saver.save(sess, os.path.join(self.check_dir, "dirac"), global_step=epoch)

    def test(self):

        if (self.do_setup):
            self.model_setup()

        if self.dataset == 'cifar-10':
            self.load_cifar_dataset('test')
            self.normalize_input(self.input_imgs)
        else:
            print('No such dataset exist')
            sys.exit()

        init = tf.global_variables_initializer()

        if not os.path.exists(self.images_dir + "/test/"):
            os.makedirs(self.images_dir + "/test/")
        if not os.path.exists(self.check_dir):
            print("No checkpoint directory exist.")
            sys.exit()

        with tf.Session() as sess:

            sess.run(init)

            if self.load_checkpoint:
                chkpt_fname = tf.train.latest_checkpoint(self.check_dir)
                saver.restore(sess, chkpt_fname)

            for itr in range(0, int(self.num_test_images / self.batch_size)):
                imgs = self.train_images[itr * self.batch_size:(itr + 1) * (self.batch_size)]
                labels = self.train_labels[itr * self.batch_size:(itr + 1) * (self.batch_size)]

                test_output = sess.run([self.final_output],
                                       feed_dict={self.input_imgs: imgs, self.input_labels: labels})

                print(test_output)
                print(labels)



if __name__ == "__main__" :

    model = UDnet()
    model.test()