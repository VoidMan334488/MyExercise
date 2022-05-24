from __future__ import print_function
import tensorflow as tf
import numpy as np
import cv2
import sys
sys.path.append("game/")
import game.wrapped_flappy_bird as game
import random

GAME = 'flappy bird'
ACTIONS = 2
GAMMA = 0.99
OBSERVE = 10000.
EXPLORE = 3.0e6
FINAL_EPSILON = 1.0e-4
INITIAL_EPSILON = 0.1
REPLAY_MEMORY = 50000
BATCH = 32
FRAME_PER_ACTION = 1

#定义经验回报类，完成数据存储和采样
class Experience_Buffer():
    def __init__(self, buffer_size = REPLAY_MEMORY):
        self.buffer = []
        self.buffer_size = buffer_size
    def add_experience(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer.size:
            self.buffer.extend(experience)
    def sample(self,samples_num):
        sample_data = random.sample(self.buffer, samples_num)
        train_a = [d[0] for d in sample_data]
        train_b = [d[1] for d in sample_data]
        train_c = [d[2] for d in sample_data]
        train_d = [d[3] for d in sample_data]
        train_terminal = [d[4] for d in sample_data]
        return train_a,train_b,train_c,train_d,train_terminal

#定义值函数网络，完成神经网络的创建和训练
class Deep_Q_N():
    def __init__(self, lr = 1.0e-6, model_file = None):
        self.gamma = GAMMA
        self.tau = 0.01
        #tf工程
        self.sess = tf.Session()
        self.learning_rate = lr
        #1.输入层
        self.obs_Q = tf.placeholder(tf.float32, shape=[None, 80, 80, 4])
        self.obs_T = tf.placeholder(tf.float32, shape=[None, 80, 80, 4])
        self.action = tf.placeholder(tf.float32, shape=[None,ACTIONS])
        #2.1 创建深度Q网络
        self.Q = self.build_q_net(self.obs_Q, scope='eval', trainable=True)
        #2.2 创建目标网络
        self.T = self.build_t_net(self.obs_T, scope='target', trainable=False)
        #2.3 整理两套网络参数
        self.qe_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval')
        self.qt_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')
        #2.4 定义新旧参数的替换操作
        self.update_old_q_op=[old_q.assign((1 - self.tau) * old_q + self.tau * p)
        for p, old_q in zip(self.qe_params, self.qt_params)]
        #3. 构建损失函数
        #td目标
        self.Q_target = tf.placeholder(tf.float32, [None])
        readout_q = tf.reduce_sum(tf.multiply(self.Q, self.action), reduction_indices=1)
        self.q_loss = tf.losses.mean_squared_error(labels=self.Q_target, predictions=readout_q)
        #4. 定义优化器
        self.q_train_op = tf.train.AdamOptimizer(lr).minimize(self.q_loss, var_list=self.qe_params)
        #5. 初始化图中变量
        self.sess.run(tf.global_variables_initializer())
        #6. 定义保存和恢复模型
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)
    #定义存储模型函数
    def save_model(self, model_path, global_step):
        self.saver.save(self.sess, model_path, global_step=global_step)
    #定义恢复模型函数
    def restore_model(self, model_path):
        self.saver.restore(self.sess, model_path)
    #创建深度q网络
    def build_q_net(self, obs, scope, trainable):
        with tf.variable_scope(scope):
            h_convl = tf.layers.conv2d(inputs=obs, filters=32, kernel_size=[8,8],
            strides=4,padding="same",activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
            bias_initializer=tf.constant_initializer(0.1),trainable=trainable)