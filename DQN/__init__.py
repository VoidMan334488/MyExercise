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


# 定义经验回报类，完成数据存储和采样
class Experience_Buffer():
    def __init__(self, buffer_size=REPLAY_MEMORY):
        self.buffer = []
        self.buffer_size = buffer_size

    def add_experience(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer.size:
            self.buffer.extend(experience)

    def sample(self, samples_num):
        sample_data = random.sample(self.buffer, samples_num)
        train_a = [d[0] for d in sample_data]
        train_b = [d[1] for d in sample_data]
        train_c = [d[2] for d in sample_data]
        train_d = [d[3] for d in sample_data]
        train_terminal = [d[4] for d in sample_data]
        return train_a, train_b, train_c, train_d, train_terminal


# 定义值函数网络，完成神经网络的创建和训练
class Deep_Q_N():
    def __init__(self, lr=1.0e-6, model_file=None):
        self.gamma = GAMMA
        self.tau = 0.01
        # tf工程
        self.sess = tf.Session()
        self.learning_rate = lr
        # 1.输入层
        self.obs_Q = tf.placeholder(tf.float32, shape=[None, 80, 80, 4])
        self.obs_T = tf.placeholder(tf.float32, shape=[None, 80, 80, 4])
        self.action = tf.placeholder(tf.float32, shape=[None, ACTIONS])
        # 2.1 创建深度Q网络
        self.Q = self.build_q_net(self.obs_Q, scope='eval', trainable=True)
        # 2.2 创建目标网络
        self.T = self.build_t_net(self.obs_T, scope='target', trainable=False)
        # 2.3 整理两套网络参数
        self.qe_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval')
        self.qt_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')
        # 2.4 定义新旧参数的替换操作
        self.update_old_q_op = [old_q.assign((1 - self.tau) * old_q + self.tau * p) for p, old_q in
                                zip(self.qe_params, self.qt_params)]
        # 3. 构建损失函数
        # td目标
        self.Q_target = tf.placeholder(tf.float32, [None])
        readout_q = tf.reduce_sum(tf.multiply(self.Q, self.action), reduction_indices=1)
        self.q_loss = tf.losses.mean_squared_error(labels=self.Q_target, predictions=readout_q)
        # 4. 定义优化器
        self.q_train_op = tf.train.AdamOptimizer(lr).minimize(self.q_loss, var_list=self.qe_params)
        # 5. 初始化图中变量
        self.sess.run(tf.global_variables_initializer())
        # 6. 定义保存和恢复模型
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)

    # 定义存储模型函数
    def save_model(self, model_path, global_step):
        self.saver.save(self.sess, model_path, global_step=global_step)

    # 定义恢复模型函数
    def restore_model(self, model_path):
        self.saver.restore(self.sess, model_path)

    # 创建深度q网络
    def build_q_net(self, obs, scope, trainable):
        with tf.variable_scope(scope):
            h_conv1 = tf.layers.conv2d(inputs=obs, filters=32, kernel_size=[8, 8], strides=4, padding="same",
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                       bias_initializer=tf.constant_initializer(0.1), trainable=trainable)
            h_pool1 = tf.layers.max_pooling2d(h_conv1, pool_size=[2, 2], strides=2, padding="same")
            h_conv2 = tf.layers.conv2d(input=h_pool1, filters=64, kernel_size=[4, 4], strides=2, padding="same",
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                       bias_initializer=tf.constant_initializer(0.1), trainable=trainable)
            h_conv3 = tf.layers.conv2d(input=h_conv2, filters=64, kernel_size=[3, 3], strides=1, padding="same",
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                       bias_initializer=tf.constant_initializer(0.1), trainable=trainable)
            h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
            # 第1个全连接层
            h_fc1 = tf.layers.dense(inputs=h_conv3_flat, units=512, activition=tf.nn.relu,
                                    kernel_initializer=tf.random_normal_initializer(0, stddev=0.1),
                                    bias_initializer=tf.constant_initializer(0.1), trainable=trainable)
            # 读出层，没有激活函数
            qout = tf.layers.dense(inputs=h_fc1, units=ACTIONS,
                                   kernel_initializer=tf.random_normal_initializer(0, stddev=0.1),
                                   bias_initialize=tf.constant_initializer(0.1), trainable=trainable)
            return qout

    def epsilon_greedy(self, s_t, epsilon):
        a_t = np.zeros([ACTIONS])
        amax = np.argmax(self.sess.run(self.Q, {self.obs_T: [s_t]})[0])
        # 概率部分
        if np.random.uniform() < 1 - epsilon:
            # 最优动作
            a_t[amax] = 1
        else:
            a_t[random.randrange(ACTIONS)] = 1
        return a_t

    def train_net(self, experience_buffer):
        # 打开游戏状态与模拟器进行通信
        game_state = game.GameState()
        # 获得第一个状态并将图像进行预处理
        do_nothing = np.zeros(ACTIONS)
        do_nothing[0] = 1
        # 与游戏交互一次
        x_t, r_0, terminal = game_state.frame_step(do_nothing)
        x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        # 开始训练
        epsilon = INITIAL_EPSILON
        t = 0
        while "flappy bird" != "angry bird":
            a_t = self.epsilon_greedy(s_t, epsilon=epsilon)
            # epsilon递减
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            # 运动动作， 与环境交互一次
            x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
            x_t1 = cv2.cvtColor(cv2.resize((x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY))
            ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
            # 将数据存储到经验池中
            experience = np.reshape(np.array([s_t, a_t, r_t, s_t1, terminal]), [1, 5])
            experience_buffer.add_experience(experience)
            # 在观察结束后进行训练
            if t > OBSERVE:
                # 采集样本
                train_s, train_a, train_r, train_s_, train_terminal = experience_buffer.sample(BATCH)
                target_q = []
                read_target_Q = self.sess.run(self.Q, {self.obs_T: train_s_})
                for i in range(len(train_r)):
                    if train_terminal[i]:
                        target_q.append(train_r[i])
                    else:
                        target_q.append(train_r[i] + GAMMA * np.max(read_target_Q[i]))
                # 训练一次
                self.sess.run(self.q_train_op,
                              feed_dict={self.obs_Q: train_s, self.action: train_a, self.Q_target: target_q})
                # 更新旧的目标网络
                self.sess.run(self.update_old_q_op)
            # 往前推进一步
            s_t = s_t1
            t += 1
            # 每一万次迭代保存一次
            if t % 10000 == 0:
                self.save_model('saved_net/', global_step=t)
            if t <= OBSERVE:
                print("OBSERVE", t)
            else:
                if t % 1 == 0:
                    print("train, steps", t, "/epsilon", epsilon, "/action_index", a_t, "/reward", r_t)


if __name__ == "__main__":
    buffer = Experience_Buffer()
    brain = Deep_Q_N()
    brain.train_net(buffer)
