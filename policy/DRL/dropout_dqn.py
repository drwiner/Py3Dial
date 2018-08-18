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
Implementation of dropout Deep Q Network

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
                    num_actor_vars, architecture = 'duel', h1_size = 130, h2_size = 50, minibatch_size=64):
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
        self.inputs, self.action, self.Qout, self.keep_prob = \
                        self.create_ddq_network(self.architecture, self.h1_size, self.h2_size)
        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_action, self.target_Qout, self.tkeep_prob = \
                        self.create_ddq_network(self.architecture, self.h1_size, self.h2_size)
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network
        self.update_target_network_params = \
            [self.target_network_params[i].assign(\
                tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.sampled_q = tf.placeholder(tf.float32, [None, 1])

        # Predicted Q given state and chosed action
        actions_one_hot = self.action
        self.pred_q = tf.reshape(tf.reduce_sum(self.Qout * actions_one_hot, axis=1, name='q_acted'),
                                 [self.minibatch_size, 1])

        # Define loss and optimization Op
        self.diff = self.sampled_q - self.pred_q
        self.loss = tf.reduce_mean(self.clipped_error(self.diff), name='loss')

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimizer.minimize(self.loss)


    def create_ddq_network(self, architecture = 'duel', h1_size = 130, h2_size = 50):
        inputs = tf.placeholder(tf.float32, [None, self.s_dim])
        action = tf.placeholder(tf.float32, [None, self.a_dim])

        W_fc1 = tf.Variable(tf.truncated_normal([self.s_dim, h1_size], stddev=0.01))
        b_fc1 = tf.Variable(tf.zeros([h1_size]))
        h_fc1 = tf.nn.relu(tf.matmul(inputs, W_fc1) + b_fc1)
        #h_fc1 = tf.nn.tanh(tf.matmul(inputs, W_fc1) + b_fc1)
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        if architecture == 'duel':

            # value function
            W_value = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.01))
            b_value = tf.Variable(tf.zeros([h2_size]))
            h_value = tf.nn.relu(tf.matmul(h_fc1, W_value) + b_value)

            W_value = tf.Variable(tf.truncated_normal([h2_size, 1], stddev=0.01))
            b_value = tf.Variable(tf.zeros([1]))
            value_out  = tf.matmul(h_value, W_value) + b_value

            # advantage function
            W_advantage = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.01))
            b_advantage = tf.Variable(tf.zeros([h2_size]))
            h_advantage = tf.nn.relu(tf.matmul(h_fc1, W_advantage) + b_advantage)

            W_advantage = tf.Variable(tf.truncated_normal([h2_size, self.a_dim], stddev=0.01))
            b_advantage = tf.Variable(tf.zeros([self.a_dim]))
            Advantage_out  = tf.matmul(h_advantage, W_advantage) + b_advantage

            Qout = value_out + (Advantage_out - tf.reduce_mean(Advantage_out, reduction_indices=1, keep_dims=True))

        else:
            W_fc2 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.01))
            b_fc2 = tf.Variable(tf.zeros([h2_size]))
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
            h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

            W_out = tf.Variable(tf.truncated_normal([h2_size, self.a_dim], stddev=0.01))
            b_out = tf.Variable(tf.zeros([self.a_dim]))
            Qout  = tf.matmul(h_fc2_drop, W_out) + b_out
            #Qout  = tf.matmul(h_fc2, W_out) + b_out

        return inputs, action, Qout, keep_prob

    def train(self, inputs, action, sampled_q, keepprob):
        return self.sess.run([self.pred_q, self.optimize, self.loss, self.diff], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.sampled_q: sampled_q,
	        self.keep_prob: keepprob,
	        self.tkeep_prob: keepprob
        })

    def predict(self, inputs, keepprob):
        return self.sess.run(self.Qout, feed_dict={
            self.inputs: inputs,
	        self.keep_prob: keepprob,
	        self.tkeep_prob: keepprob
        })

    def predict_target(self, inputs, keepprob):
        return self.sess.run(self.target_Qout, feed_dict={
            self.target_inputs: inputs,
	        self.keep_prob: keepprob,
	        self.tkeep_prob: keepprob
        })

    def predict_target_with_action_maxQ(self, inputs):
        return self.sess.run(self.action_maxQ_target, feed_dict={
            self.target_inputs: inputs,
            self.inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def load_network(self, load_filename):
        self.saver = tf.train.Saver()
        try:
            self.saver.restore(self.sess, './' +  load_filename)
            print "Successfully loaded:", load_filename
        except:
            print "Could not find old network weights"

    def save_network(self, save_filename):
        print 'Saving deepq-network...'
        self.saver.save(self.sess, './' + save_filename)

    def clipped_error(self, x):
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5) # condition, true, false
