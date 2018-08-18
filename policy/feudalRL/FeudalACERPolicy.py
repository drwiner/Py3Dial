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

'''
ACERPolicy.py - ACER - Actor Critic with Experience Replay
==================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies:

    import :class:`Policy`
    import :class:`utils.ContextLogger`

.. warning::
        Documentation not done.


************************

'''
import copy
import os
import json
import numpy as np
import scipy
import scipy.signal
import cPickle as pickle
import random
import utils
from utils.Settings import config as cfg
from utils import ContextLogger, DiaAct

import ontology.FlatOntologyManager as FlatOnt
import tensorflow as tf
from policy.DRL.replay_buffer_episode_acer import ReplayBufferEpisode
from policy.DRL.replay_prioritised_episode import ReplayPrioritisedEpisode
import policy.DRL.utils as drlutils
from policy.ACERPolicy import ACERPolicy
import policy.DRL.acer as acer
import policy.Policy
import policy.SummaryAction
from policy.Policy import TerminalAction, TerminalState
from policy.feudalRL.DIP_parametrisation import DIP_state, padded_state

logger = utils.ContextLogger.getLogger('')

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class FeudalACERPolicy(ACERPolicy):
    '''Derived from :class:`Policy`
    '''
    def __init__(self, in_policy_file, out_policy_file, domainString='CamRestaurants', is_training=False, action_names=None, slot=None):
        super(FeudalACERPolicy, self).__init__(in_policy_file, out_policy_file, domainString, is_training)

        tf.reset_default_graph()

        self.in_policy_file = in_policy_file
        self.out_policy_file = out_policy_file
        self.is_training = is_training
        self.accum_belief = []
        self.prev_state_check = None

        self.domainString = domainString
        self.domainUtil = FlatOnt.FlatDomainOntology(self.domainString)

        self.features = 'dip'
        self.sd_enc_size = 80
        self.si_enc_size = 40
        self.dropout_rate = 0.
        if cfg.has_option('feudalpolicy', 'features'):
            self.features = cfg.get('feudalpolicy', 'features')
        if cfg.has_option('feudalpolicy', 'sd_enc_size'):
            self.sd_enc_size = cfg.getint('feudalpolicy', 'sd_enc_size')
        if cfg.has_option('feudalpolicy', 'si_enc_size'):
            self.si_enc_size = cfg.getint('feudalpolicy', 'si_enc_size')
        if cfg.has_option('dqnpolicy', 'dropout_rate') and self.is_training:
            self.dropout_rate = cfg.getfloat('feudalpolicy', 'dropout_rate')
        if cfg.has_option('dqnpolicy', 'dropout_rate') and self.is_training:
            self.dropout_rate = cfg.getfloat('feudalpolicy', 'dropout_rate')
        self.actfreq_ds = False
        if cfg.has_option('feudalpolicy', 'actfreq_ds'):
            self.actfreq_ds = cfg.getboolean('feudalpolicy', 'actfreq_ds')

        # init session
        self.sess = tf.Session()
        with tf.device("/cpu:0"):

            np.random.seed(self.randomseed)
            tf.set_random_seed(self.randomseed)

            # initialise an replay buffer
            if self.replay_type == 'vanilla':
                self.episodes[self.domainString] = ReplayBufferEpisode(self.capacity, self.minibatch_size, self.randomseed)
            elif self.replay_type == 'prioritized':
                self.episodes[self.domainString] = ReplayPrioritisedEpisode(self.capacity, self.minibatch_size, self.randomseed)
            #replay_buffer = ReplayBuffer(self.capacity, self.randomseed)
            #self.episodes = []
            self.samplecount = 0
            self.episodecount = 0

            # construct the models
            self.state_dim = 89  # current DIP state dim
            self.summaryaction = policy.SummaryAction.SummaryAction(domainString)
            self.action_names = action_names
            self.action_dim = len(self.action_names)
            action_bound = len(self.action_names)
            self.stats = [0 for _ in range(self.action_dim)]

            self.global_mu = [0. for _ in range(self.action_dim)]

            if self.features == 'dip':
                if self.actfreq_ds:
                    if self.domainString == 'CamRestaurants':
                        self.state_dim += 9#16
                    elif self.domainString == 'SFRestaurants':
                        self.state_dim += 9#25
                    elif self.domainString == 'Laptops11':
                        self.state_dim += 9#40
                self.acer = acer.ACERNetwork(self.sess, self.state_dim, self.action_dim, self.critic_lr, self.delta,
                                             self.c, self.alpha, self.h1_size, self.h2_size, self.is_training)
            elif self.features == 'learned' or self.features == 'rnn':
                si_state_dim = 72
                if self.actfreq_ds:
                    if self.domainString == 'CamRestaurants':
                        si_state_dim += 9#16
                    elif self.domainString == 'SFRestaurants':
                        si_state_dim += 9#25
                    elif self.domainString == 'Laptops11':
                        si_state_dim += 9#40
                if self.domainString == 'CamRestaurants':
                    sd_state_dim = 158#94
                elif self.domainString == 'SFRestaurants':
                    sd_state_dim = 158
                elif self.domainString == 'Laptops11':
                    sd_state_dim = 158#13
                else:
                    logger.error(
                        'Domain {} not implemented in feudal-DQN yet')  # just find out the size of sd_state_dim for the new domain
                if 0:#self.features == 'rnn':
                    self.acer = acer.RNNACERNetwork(self.sess, si_state_dim, sd_state_dim, self.action_dim, self.critic_lr,
                                                    self.delta, self.c, self.alpha, self.h1_size, self.h2_size, self.is_training,
                                                    sd_enc_size=25, si_enc_size=25, dropout_rate=0., tn='normal', slot='si')
                else:
                    self.state_dim = si_state_dim + sd_state_dim
                    self.acer = acer.ACERNetwork(self.sess, self.state_dim, self.action_dim,
                                                 self.critic_lr, self.delta, self.c, self.alpha, self.h1_size,
                                                 self.h2_size, self.is_training)

            else:
                logger.error('features "{}" not implemented'.format(self.features))


            # when all models are defined, init all variables
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

            self.loadPolicy(self.in_policy_file)
            print 'loaded replay size: ', self.episodes[self.domainString].size()

            #self.acer.update_target_network()

    # def record() has been handled...

    def convertStateAction(self, state, action):
        '''

        '''
        if isinstance(state, TerminalState):
            return [0] * 89, action

        else:
            if self.features == 'learned' or self.features == 'rnn':
                dip_state = padded_state(state.domainStates[state.currentdomain], self.domainString)
            else:
                dip_state = DIP_state(state.domainStates[state.currentdomain], self.domainString)
            action_name = self.actions.action_names[action]
            act_slot = 'general'
            for slot in dip_state.slots:
                if slot in action_name:
                    act_slot = slot
            flat_belief = dip_state.get_beliefStateVec(act_slot)
            self.prev_state_check = flat_belief

            return flat_belief, action

    def record(self, reward, domainInControl=None, weight=None, state=None, action=None):
        if domainInControl is None:
            domainInControl = self.domainString
        if self.actToBeRecorded is None:
            self.actToBeRecorded = self.summaryAct

        if state is None:
            state = self.prevbelief
        if action is None:
            action = self.actToBeRecorded
        mu_weight = self.prev_mu
        mask = self.prev_mask
        if action == self.action_dim-1: # pass action was taken
            mask = np.zeros(self.action_dim)
            mu_weight = np.ones(self.action_dim)/self.action_dim

        cState, cAction = state, action

        reward /= 20.0

        value = self.acer.predict_value([cState], [mask])

        if self.replay_type == 'vanilla':
            self.episodes[domainInControl].record(state=cState, \
                    state_ori=state, action=cAction, reward=reward, value=value[0], distribution=mu_weight, mask=mask)
        elif self.replay_type == 'prioritized':
            self.episodes[domainInControl].record(state=cState, \
                    state_ori=state, action=cAction, reward=reward, value=value[0], distribution=mu_weight, mask=mask)

        self.actToBeRecorded = None
        self.samplecount += 1
        return

    def finalizeRecord(self, reward, domainInControl=None):
        if domainInControl is None:
            domainInControl = self.domainString
        if self.episodes[domainInControl] is None:
            logger.warning("record attempted to be finalized for domain where nothing has been recorded before")
            return

        #print 'Episode Avg_Max_Q', float(self.episode_ave_max_q)/float(self.episodes[domainInControl].size())
        #print 'Episode Avg_Max_Q', np.mean(self.episode_ave_max_q)
        #print self.stats

        # normalising total return to -1~1
        reward /= 20.0

        terminal_state, terminal_action = self.convertStateAction(TerminalState(), TerminalAction())
        value = 0.0 # not effect on experience replay

        def calculate_discountR_advantage(r_episode, v_episode):
            #########################################################################
            # Here we take the rewards and values from the rollout, and use them to
            # generate the advantage and discounted returns.
            # The advantage function uses "Generalized Advantage Estimation"
            bootstrap_value = 0.0
            self.r_episode_plus = np.asarray(r_episode + [bootstrap_value])
            discounted_r_episode = discount(self.r_episode_plus,self.gamma)[:-1]
            self.v_episode_plus = np.asarray(v_episode + [bootstrap_value])
            advantage = r_episode + self.gamma * self.v_episode_plus[1:] - self.v_episode_plus[:-1]
            advantage = discount(advantage,self.gamma)
            #########################################################################
            return discounted_r_episode, advantage

        if self.replay_type == 'vanilla':
            self.episodes[domainInControl].record(state=terminal_state, \
                    state_ori=TerminalState(), action=terminal_action, reward=reward, value=value, terminal=True, distribution=None)
        elif self.replay_type == 'prioritized':
            episode_r, episode_v = self.episodes[domainInControl].record_final_and_get_episode(state=terminal_state, \
                                                                                               state_ori=TerminalState(),
                                                                                               action=terminal_action,
                                                                                               reward=reward,
                                                                                               value=value)

            # TD_error is a list of td error in the current episode
            _, TD_error = calculate_discountR_advantage(episode_r, episode_v)
            episodic_TD = np.mean(np.absolute(TD_error))
            print 'episodic_TD'
            print episodic_TD
            self.episodes[domainInControl].insertPriority(episodic_TD)

        return


    def nextAction(self, beliefstate):
        '''
        select next action

        :param beliefstate:
        :param hyps:
        :returns: (int) next summarye action
        '''

        #execMask = self.summaryaction.getExecutableMask(beliefstate, self.lastSystemAction)
        execMask = np.zeros(self.action_dim)

        def apply_mask(prob, maskval, baseline=9.99999975e-06):
            return prob if maskval == 0.0 else baseline # not quite 0.0 to avoid division by zero


        if self.exploration_type == 'e-greedy' or not self.is_training:
            if self.is_training and utils.Settings.random.rand() < self.epsilon:
                action_prob = np.random.rand(len(self.action_names))
            else:
                action_prob = self.acer.predict_policy(np.reshape(beliefstate, (1, len(beliefstate))),
                                                   np.reshape(execMask, (1, len(execMask))))[0]
        mu = action_prob / sum(action_prob)
        self.prev_mu = mu
        self.prev_mask = execMask
        return action_prob

    def train(self):
        '''
        call this function when the episode ends
        '''
        USE_GLOBAL_MU = False
        self.episode_ct += 1

        if not self.is_training:
            logger.info("Not in training mode")
            return
        else:
            logger.info("Update acer policy parameters.")

        self.episodecount += 1
        logger.info("Sample Num so far: %s" % (self.samplecount))
        logger.info("Episode Num so far: %s" % (self.episodecount))
        if self.samplecount >= self.minibatch_size * 3 and self.episodecount % self.training_frequency == 0:
        #if self.episodecount % self.training_frequency == 0:
            logger.info('start trainig...')

            for _ in range(self.train_iters_per_episode):


                if self.replay_type == 'vanilla' or self.replay_type == 'prioritized':
                    s_batch, s_ori_batch, a_batch, r_batch, s2_batch, s2_ori_batch, t_batch, idx_batch, v_batch, mu_policy, mask_batch = \
                        self.episodes[self.domainString].sample_batch()
                    if USE_GLOBAL_MU:
                        mu_sum = sum(self.global_mu)
                        mu_normalised = np.array([c / mu_sum for c in self.global_mu])
                        mu_policy = [[mu_normalised for _ in range(len(mu_policy[i]))] for i in range(len(mu_policy))]
                else:
                    assert False  # not implemented yet

                discounted_r_batch = []
                advantage_batch = []
                def calculate_discountR_advantage(r_episode, v_episode):
                    #########################################################################
                    # Here we take the rewards and values from the rolloutv, and use them to
                    # generate the advantage and discounted returns.
                    # The advantage function uses "Generalized Advantage Estimation"
                    bootstrap_value = 0.0
                    # r_episode rescale by rhos?
                    self.r_episode_plus = np.asarray(r_episode + [bootstrap_value])
                    discounted_r_episode = discount(self.r_episode_plus, self.gamma)[:-1]
                    self.v_episode_plus = np.asarray(v_episode + [bootstrap_value])
                    # change sth here
                    advantage = r_episode + self.gamma * self.v_episode_plus[1:] - self.v_episode_plus[:-1]
                    advantage = discount(advantage, self.gamma)
                    #########################################################################
                    return discounted_r_episode, advantage

                if self.replay_type == 'prioritized':
                    for item_r, item_v, item_idx in zip(r_batch, v_batch, idx_batch):
                        # r, a = calculate_discountR_advantage(item_r, np.concatenate(item_v).ravel().tolist())
                        r, a = calculate_discountR_advantage(item_r, item_v)

                        # flatten nested numpy array and turn it into list
                        discounted_r_batch += r.tolist()
                        advantage_batch += a.tolist()

                        # update the sum-tree
                        # update the TD error of the samples (episode) in the minibatch
                        episodic_TD_error = np.mean(np.absolute(a))
                        self.episodes[self.domainString].update(item_idx, episodic_TD_error)
                else:
                    for item_r, item_v in zip(r_batch, v_batch):
                        # r, a = calculate_discountR_advantage(item_r, np.concatenate(item_v).ravel().tolist())
                        r, a = calculate_discountR_advantage(item_r, item_v)

                        # flatten nested numpy array and turn it into list
                        discounted_r_batch += r.tolist()
                        advantage_batch += a.tolist()

                batch_size = len(s_batch)

                a_batch_one_hot = np.eye(self.action_dim)[np.concatenate(a_batch, axis=0).tolist()]

                loss, entropy, optimize = \
                            self.acer.train(np.concatenate(np.array(s_batch), axis=0).tolist(), a_batch_one_hot,
                                            np.concatenate(np.array(mask_batch), axis=0).tolist(),
                                            np.concatenate(np.array(r_batch), axis=0).tolist(), s_batch, r_batch, self.gamma,
                                            np.concatenate(np.array(mu_policy), axis=0),
                                            discounted_r_batch, advantage_batch)

                ent, norm_loss = entropy/float(batch_size), loss/float(batch_size)


            self.savePolicyInc()  # self.out_policy_file)


    def savePolicy(self, FORCE_SAVE=False):
        """
        Does not use this, cause it will be called from agent after every episode.
        we want to save the policy only periodically.
        """
        pass

    def savePolicyInc(self, FORCE_SAVE=False):
        """
        save model and replay buffer
        """
        if self.episodecount % self.save_step == 0:
            #save_path = self.saver.save(self.sess, self.out_policy_file+'.ckpt')
            self.acer.save_network(self.out_policy_file+'.acer.ckpt')

            f = open(self.out_policy_file+'.episode', 'wb')
            for obj in [self.samplecount, self.episodes[self.domainString], self.global_mu]:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
            #logger.info("Saving model to %s and replay buffer..." % save_path)

    def loadPolicy(self, filename):
        """
        load model and replay buffer
        """
        # load models
        self.acer.load_network(filename+'.acer.ckpt')

        # load replay buffer
        try:
            print 'load from: ', filename
            f = open(filename+'.episode', 'rb')
            loaded_objects = []
            for i in range(2): # load nn params and collected data
                loaded_objects.append(pickle.load(f))
            self.samplecount = int(loaded_objects[0])
            self.episodes[self.domainString] = copy.deepcopy(loaded_objects[1])
            self.global_mu = loaded_objects[2]
            logger.info("Loading both model from %s and replay buffer..." % filename)
            f.close()
        except:
            logger.info("Loading only models...")

    def restart(self):
        self.summaryAct = None
        self.lastSystemAction = None
        self.prevbelief = None
        self.prev_mu = None
        self.prev_mask = None
        self.actToBeRecorded = None
        self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * float(self.episodeNum+self.episodecount) / float(self.maxiter)
        self.episode_ave_max_q = []

#END OF FILE
