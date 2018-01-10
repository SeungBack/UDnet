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
import logging
import numpy as np
import tifffile as tiff
import sys
import cv2 as cv
sys.path.append('/home/seung/UDnet/isbi-2012/')

from basic_layers import *
from dirac_layers import *
from layers import *
from isbi_utils import *
from math import *

'''
hyperparameters

batch_norm decay : 
Decay for the moving average. Reasonable values for decay are close to 1.0, 
typically in the multiple-nines ran/ge: 0.999, 0.99, 0.9, etc. Lower decay value (recommend trying decay=0.9) 
if model experiences reasonably good training performance but poor validation and/or test performance. 
Try zero_debias_moving_mean=True for improved stability.

'''

'''
augmentation

model change
'''

class UDnet():

    def run_parser(self):
        self.parser = optparse.OptionParser()

        self.parser.add_option('--num_iter', type='int', default=1000, dest='num_iter')
        self.parser.add_option('--batch_size', type='int', default=3, dest='batch_size')
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
        self.parser.add_option('--lr', type='float', default=0.1, dest='lr')
        self.parser.add_option('--rf', type='float', default=0.00005, dest='rf')

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
        self.lr = opt.lr
        self.rf = opt.rf

    def load_isbi_dataset(self, mode='train'):

        if self.dataset == 'isbi':

            if (mode == 'train'):

                imgs_path = '/home/seung/UDnet/ISBI/train-volume.tif'
                msks_path = '/home/seung/UDnet/ISBI/train-labels.tif'

                imgs, msks = tiff.imread(imgs_path) / 255, tiff.imread(msks_path) / 255

                self.train_images, self.train_labels = unison_shuffled_copies(imgs, msks)
                self.train_images = np.reshape(self.train_images,
                                          [self.num_images, self.img_height, self.img_width, self.img_depth])
                self.train_labels = np.reshape(self.train_labels,
                                          [self.num_images, self.img_height, self.img_width, self.img_depth])
                self.sample_imgs = self.train_images[0:self.batch_size]

            elif (mode == 'test'):

                test_imgs_path='/home/seung/UDnet/ISBI/test-volume.tif'
                self.test_images = tiff.imread(test_imgs_path) /255
                self.test_images = np.reshape(self.test_images,[self.num_images, self.img_height, self.img_width, self.img_depth])
        else:
            print("Model not supported for this dataset")
            sys.exit()

    def model_setup(self):

        with tf.variable_scope("Model") as scope:

            self.input_imgs = tf.placeholder(tf.float32,
                                             shape = [self.batch_size, self.img_height, self.img_width, self.img_depth])
            self.input_labels = tf.placeholder(tf.int32,
                                             shape = [self.batch_size, self.img_height, self.img_width, self.img_depth])
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




    def isbi_model_setup(self):

        self.input_imgs = tf.placeholder(tf.float32,[self.batch_size, self.img_height, self.img_width, self.img_depth]) # 512x512x1
        self.input_labels = tf.placeholder(tf.int32,
                                           shape=[self.batch_size, self.img_height, self.img_width, self.img_depth])

        long_skip = []

        ################ encoding path ##############
        for group in range(0, self.num_groups):

            if group == 0:
                dim = 64
                conv = self.input_imgs
            else:
                dim = dim*2
            short_skip = dirac_conv2d(conv, output_dim=dim, filter_height=3, filter_width=3, stride_height=1,
                                stride_width=1,stddev=0.02, padding="SAME", do_norm=True, norm_type='batch_norm',
                                name="encode_diracconv_" + str(group) +"_1")
            print(short_skip)
            conv = residual_block(inputconv = short_skip , num_iter=3,  output_dim=dim, filter_height=3, filter_width=3,
                                  stride_height=1, stride_width=1, stddev=0.02, padding="SAME", do_norm=True,
                                  norm_type='batch_norm', dropout=self.dropout, name = "encode_res_block_" + str(group))
            conv = conv + short_skip
            print(conv)
            conv = dirac_conv2d(conv, output_dim=dim, filter_height=3, filter_width=3, stride_height=1,
                                stride_width=1,stddev=0.02, padding="SAME", do_norm=True, norm_type='batch_norm',
                                name="encode_diracconv_" + str(group) + "_2")
            print(conv)
            long_skip.append(conv)
            conv = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name="encode_max_" + str(group))
            print("max : ", conv)
        ################ bottle neck ################
        dim = dim*2
        conv = dirac_conv2d(conv, output_dim=dim, filter_height=3, filter_width=3, stride_height=1,
                            stride_width=1, stddev=0.02, padding="SAME", do_norm=True, norm_type='batch_norm',
                            name="dirac_conv2d_bottleneck_1")
        print(conv)
        conv = conv + residual_block(inputconv=conv, num_iter=3, output_dim=dim, filter_height=3, filter_width=3,
                              stride_height=1, stride_width=1, stddev=0.02, padding="SAME", do_norm=True,
                              norm_type='batch_norm', dropout=self.dropout, name="bottleneck_res_block_" + str(group))
        print(conv)
        conv = dirac_conv2d(conv, output_dim=dim, filter_height=3, filter_width=3, stride_height=1,
                            stride_width=1, stddev=0.02, padding="SAME", do_norm=True, norm_type='batch_norm',
                            name="dirac_conv2d_bottleneck_2")
        print(conv)


        ################ decoding path ################
        for group in range(0, self.num_groups):

            dim = int(dim/2)
            conv = tf.contrib.layers.conv2d_transpose(inputs = conv, num_outputs = dim, kernel_size = 2, stride = 2)
            print(conv)
            conv = conv + long_skip[self.num_groups - group -1]

            conv = dirac_conv2d(conv, output_dim=dim, filter_height=3, filter_width=3,
                                stride_height=1, stride_width=1, stddev=0.02, padding="SAME", do_norm=True,
                                norm_type='batch_norm', name="decode_diracconv_" + str(group) +"_1")

            conv = conv + residual_block(inputconv=conv, num_iter=3, output_dim=dim, filter_height=3,
                                  filter_width=3, stride_height=1, stride_width=1, stddev=0.02, padding="SAME",
                                  do_norm=True, norm_type='batch_norm', dropout=self.dropout, name = "decode_res_block_" + str(group))

            conv = dirac_conv2d(conv, output_dim=dim, filter_height=3, filter_width=3, stride_height=1,
                                stride_width=1, stddev=0.02, padding="SAME", do_norm=True, norm_type='batch_norm',
                                name="decode_diracconv_" + str(group) + "_2")


        ################ final output ###############
        conv = tf.contrib.layers.conv2d(conv, 1, [1,1], [1,1], padding='SAME', activation_fn=None,
                                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),biases_initializer=tf.constant_initializer(0.0))

        conv = tf.nn.relu(conv, "final_relu")
        self.final_output = tf.sigmoid(conv, name="final_sigmoid")

        #temp_shape = conv.get_shape().as_list()
        #o_avgpool = tf.nn.avg_pool(conv, [1, temp_shape[1], temp_shape[2], 1], [1, temp_shape[1], temp_shape[2], 1],
        #                           "VALID", name="avgpool")
        #temp_depth = o_avgpool.get_shape().as_list()[-1]
        #self.final_output = linear1d(tf.reshape(o_avgpool, [self.batch_size, temp_depth]), temp_depth, 10)


        # o_simple1 = self.simple_block(o_ctop, 32, 3, 3, 1, 1, maxpool=True, name="simple_down") #256x256x32
        # o_bottleneck1 = self.bottleneck() #128x128x128
        #
        # for group in range(0, self.num_groups):
        #     for block in range(0, self.num_blocks):
        #         o_loop = ncrelu(o_loop, name="crelu_" + str(group) + "_" + str(block))
        #         o_loop = dirac_conv2d(o_loop, outdim, 3, 3, 1, 1, name="conv_" + str(group) + "_" + str(block))
        #
        #     if (group != self.num_groups - 1):
        #         o_loop = tf.nn.pool(o_loop, [2, 2], "MAX", "VALID", None, [2, 2], name="maxpool_" + str(group))
        #
        #     outdim = outdim * 2
        #
        # temp_shape = o_loop.get_shape().as_list()
        # o_avgpool = tf.nn.avg_pool(o_loop, [1, temp_shape[1], temp_shape[2], 1], [1, temp_shape[1], temp_shape[2], 1],
        #                            "VALID", name="avgpool")
        # temp_depth = o_avgpool.get_shape().as_list()[-1]
        # self.final_output = linear1d(tf.reshape(o_avgpool, [self.batch_size, temp_depth]), temp_depth, 10)

    def loss_setup(self):

        #self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_labels, logits=self.final_output, name="Error_loss")
        self.loss = tf.losses.mean_squared_error(labels=self.input_labels, predictions=self.final_output)
        self.loss = self.loss + tf.add_n([tf.nn.l2_loss(v) for v in self.model_vars]) * self.rf # for regularization
        self.loss = tf.reduce_mean(self.loss)


        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)
        #optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
        self.loss_optimizer = optimizer.minimize(self.loss)

        # Defining the summary ops
        self.cl_loss_summ = tf.summary.scalar("cl_loss", self.loss)


    def train(self):

        self.model_setup()
        self.loss_setup()

        #if self.dataset == 'isbi' :
        self.load_isbi_dataset('train')
        #self.normalize_input(self.input_imgs)

     #   else:
     #       print('No such dataset exist')
     #       sys.exit()

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        if not os.path.exists(self.images_dir + "/train/"):
            os.makedirs(self.images_dir + "/train/")
        if not os.path.exists(self.check_dir):
            os.makedirs(self.check_dir)

        with tf.Session() as sess:

            print("-----------------start learning--------------")
            sess.run(init)
            writer = tf.summary.FileWriter(self.tensorboard_dir)
            writer.add_graph(sess.graph)

            if self.load_checkpoint:
                chkpt_fname = tf.train.latest_checkpoint(self.check_dir)
                saver.restore(sess, chkpt_fname)
                print("loaded checkpoint sucessfully")

            for epoch in range(0, self.max_epoch):

                self.train_images, self.train_labels = unison_shuffled_copies(self.train_images, self.train_labels)
                if epoch % 20 == 0:
                    self.lr = 0.95*self.lr
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

                output = sess.run([self.final_output], feed_dict={self.input_imgs: self.sample_imgs})
                result = output[0][0]
                result = np.reshape(result, [self.img_width, self.img_height])
                result = np.round(result)
                result = result * 255.0
                cv.imwrite(self.images_dir+'/result_epoch_'+str(epoch+1)+'.png', result)



        print("learning finish")

    def normalize_input(self, imgs):

        return imgs / 127.5 - 1.0

    def test(self):

        if (self.do_setup):
            self.model_setup()

        if self.dataset == 'cifar-10':
            self.load_cifar_dataset('test')
            #self.normalize_input(self.input_imgs)
        if self.dataset == 'isbi':
            self.load_isbi_dataset('test')
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
                imgs = self.test_images[itr * self.batch_size:(itr + 1) * (self.batch_size)]
                test_output = sess.run([self.final_output], feed_dict={self.input_imgs: imgs})



                if (itr == self.max_epoch - 1):
                    result = sess.run([self.final_output], feed_dict={self.input_imgs: self.train_images})
                    result = np.reshape(result, [self.num_images, self.img_width, self.img_height])
                    tiff.imsave(self.images_dir + 'final_tr_result_.tif', result)

                else:
                    result = test_output[0][0]
                    result = np.reshape(result, [self.img_width, self.img_height])
                    result = result * 255.0
                    cv.imwrite('./result.png', result)

            result = sess.run([self.final_output], feed_dict={self.input_imgs: self.test_images})
            result = np.reshape(result, [self.num_images, self.img_width, self.img_height])
            result = np.round(result)
            tiff.imsave(self.images_dir+'final_test_result_.tif', result)

        print("test finish")


if __name__ == "__main__" :

    model = UDnet()
    if (model.to_test):
        model.test()
    else:
        model.train()