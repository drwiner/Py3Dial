###############################################################################
# PyDial: Multi-domain Statistical Spoken Dialogue System Software
###############################################################################
#
# Copyright 2015 - 2018
# Cambridge University Engineering Department Dialogue Systems Group
#
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
###############################################################################

""" 
Implementation of Bootstrapped Deep Q Network

The algorithm is developed with Tensorflow

Copyright CUED Dialogue Systems Group 2015 - 2018
"""
import tensorflow as tf

# ===========================
#   Deep Q Network
# ===========================
class DeepQNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    """
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, \
                    num_actor_vars, architecture='duel', h1_size=130, h2_size=50, no_head=1, minibatch_size=64):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.architecture = architecture
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.minibatch_size = minibatch_size

        # Create the deep Q network
        self.inputs, self.action, self.hidden = self.create_ddq_trunk(self.h1_size, self.h2_size)

        self.heads = []
        for head in range(no_head):
            self.heads.append(self.create_ddq_head(self.hidden, h1_size=130, h2_size=50))

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_action, self.target_hidden = self.create_ddq_trunk(self.h1_size, self.h2_size)

        self.target_heads = []
        for head in range(no_head):
            self.target_heads.append(self.create_ddq_head(self.target_hidden, h1_size=130, h2_size=50))

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network
        self.update_target_network_params = \
            [self.target_network_params[i].assign(\
                tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.sampled_q = tf.placeholder(tf.float32, [None, 1])

        # Predicted Q given state and chosed action
        #actions_one_hot = tf.one_hot(self.action, self.a_dim, 1.0, 0.0, name='action_one_hot')
        actions_one_hot = self.action

        # predictions
        self.pred_q = []
        for head in range(no_head):
            # self.pred_q.append(tf.reshape(tf.reduce_sum(self.heads[head] * actions_one_hot,
            #                                             reduction_indices=1, name='q_acted' + str(head)), [self.minibatch_size, 1]))

            self.pred_q.append(tf.reshape(tf.reduce_sum(self.heads[head] * actions_one_hot, axis=1, name='q_acted'),
                                          [self.minibatch_size, 1]))

        # Define loss and optimization Op
        self.diff = []
        for head in range(no_head):
            self.diff.append(self.sampled_q - self.pred_q[head])

        self.loss = []
        for head in range(no_head):
            self.loss.append(tf.reduce_mean(self.clipped_error(self.diff[head]), name='loss'))

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        self.optimize = []
        for head in range(no_head):
            self.optimize.append(self.optimizer.minimize(self.loss[head]))

    def create_ddq_trunk(self, h1_size = 130, h2_size = 50):
        inputs = tf.placeholder(tf.float32, [None, self.s_dim])
        action = tf.placeholder(tf.float32, [None, self.a_dim])

        W_fc1 = tf.Variable(tf.truncated_normal([self.s_dim, h1_size], stddev=0.01))
        b_fc1 = tf.Variable(tf.zeros([h1_size]))
        h_fc1 = tf.nn.relu(tf.matmul(inputs, W_fc1) + b_fc1)

        W_fc2 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.01))
        b_fc2 = tf.Variable(tf.zeros([h2_size]))
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

        return inputs, action, h_fc2


    def create_ddq_head(self, hidden, h1_size = 130, h2_size = 50):
        # creation of new head
        W_out = tf.Variable(tf.truncated_normal([h2_size, self.a_dim], stddev=0.01))
        b_out = tf.Variable(tf.zeros([self.a_dim]))
        Qout  = tf.matmul(hidden, W_out) + b_out

        return Qout

    def create_target_network(self, architecture = 'duel', h1_size = 130, h2_size = 50):
        inputs = tf.placeholder(tf.float32, [None, self.s_dim])
        action = tf.placeholder(tf.float32, [None, self.a_dim])

        W_fc1 = tf.Variable(tf.truncated_normal([self.s_dim, h1_size], stddev=0.01))
        b_fc1 = tf.Variable(tf.zeros([h1_size]))
        h_fc1 = tf.nn.relu(tf.matmul(inputs, W_fc1) + b_fc1)

        W_fc2 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.01))
        b_fc2 = tf.Variable(tf.zeros([h2_size]))
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        #h_fc1 = tf.nn.tanh(tf.matmul(inputs, W_fc1) + b_fc1)

        W_out = tf.Variable(tf.truncated_normal([h2_size, self.a_dim], stddev=0.01))
        b_out = tf.Variable(tf.zeros([self.a_dim]))
        Qout  = tf.matmul(h_fc2, W_out) + b_out

        return inputs, action, Qout

    def train(self, inputs, action, sampled_q, head):
        return self.sess.run([self.pred_q[head], self.optimize[head], self.loss[head], self.diff[head]], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.sampled_q: sampled_q
        })

    def predict(self, inputs, head):
        return self.sess.run(self.heads[head], feed_dict={
            self.inputs: inputs
        })

    def predict_action(self, inputs, head):
        return self.sess.run(self.pred_q[head], feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs, head):
        return self.sess.run(self.target_heads[head], feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self, head):
        self.sess.run(self.update_target_network_params[head])

    def load_network(self, load_filename):
        self.saver = tf.train.Saver()
        try:
            self.saver.restore(self.sess, load_filename)
            print "Successfully loaded:", './' + load_filename
        except:
            print "Could not find old network weights"

    def save_network(self, save_filename):
        print 'Saving deepq-network...'
        self.saver.save(self.sess, './' + save_filename)

    def clipped_error(self, x):
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5) # condition, true, false
