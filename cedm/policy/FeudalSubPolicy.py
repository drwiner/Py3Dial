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

from copy import deepcopy
import SummaryActionRelOnly
from cedm.utils import DActEntity
import SummaryAction
import policy.GPPolicy
from policy.GPLib import GPSARSA
from utils import Settings, ContextLogger, DiaAct
import math, scipy.stats
from ontology import Ontology
logger = ContextLogger.getLogger('')

class FeudalGPSubPolicy(policy.GPPolicy.GPPolicy):
    
    def __init__(self,domainString, learning, sharedParams=None, rolename = None):
        super(FeudalGPSubPolicy, self).__init__(domainString, learning, sharedParams)
        if rolename is not None:
            inpolicyfile = ''
            outpolicyfile = ''
            
            if Settings.config.has_option('policy', 'inpolicyfile'):
                inpolicyfile = Settings.config.get('policy', 'inpolicyfile')
            if Settings.config.has_option('policy', 'outpolicyfile'):
                outpolicyfile = Settings.config.get('policy', 'outpolicyfile')
            if Settings.config.has_option('policy_'+domainString, 'inpolicyfile'):
                inpolicyfile = Settings.config.get('policy_'+domainString, 'inpolicyfile')
            if Settings.config.has_option('policy_'+domainString, 'outpolicyfile'):
                outpolicyfile = Settings.config.get('policy_'+domainString, 'outpolicyfile')
            
            outpolicyfile += "_{}".format(rolename)
            inpolicyfile += "_{}".format(rolename)
            # Learning algorithm:
            self.learner = GPSARSA(inpolicyfile,outpolicyfile,domainString=domainString, learning=self.learning, sharedParams=sharedParams)
        
        self.rolename = rolename
        
    
    def peekPolicy(self, belief):
        nonExecutableActions = self.actions.getNonExecutable(belief, self.lastSystemAction)
        if self._byeAction is not None:
            nonExecutableActions.append(self._byeAction)

        currentstate = self.get_State(belief)
        executable = self._createExecutable(nonExecutableActions)
        
#         print "-------------- non-executable actions: {}".format(nonExecutableActions)
#         print "--------------     executable actions: "
#         for act in executable:
#             print act.toString()
#         
#         print "################## GPState"
#         print currentstate._bstate

        if len(executable) < 1:
            logger.error("No executable actions")

         
        
        if self._byeAction is not None:
            nonExecutableActions.append(self._byeAction)
        
        executable = self._createExecutable(nonExecutableActions) # own domains abstracted actions
        
        state = currentstate
        kernel = self.kernel
        
        Q =[]
        for action in executable:
            if self._scale <= 0:
                [mean, var] = self.QvalueMeanVar(state, action, kernel)
                logger.debug('action: ' +str(action.act) + ' mean then var:\t\t\t ' + str(mean) + '  ' + str(math.sqrt(var)))
                value = mean
                gaussvar = 0
            else:
                [mean, var ] = self.QvalueMeanVar(state, action, kernel)                
                gaussvar = self._scale * math.sqrt(var)                        
                value = gaussvar * Settings.random.randn() + mean     # Sample a Q value for this action
                logger.debug('action: ' +str(action.act) + ' mean then var:\t\t\t ' + str(mean) + '  ' + str(gaussvar))
            Q.append((action, value, mean, gaussvar))

        Q=sorted(Q,key=lambda Qvalue : Qvalue[1], reverse=True)
        
        best_action, best_actions_sampled_Q_value = Q[0][0], Q[0][1]
        actions_likelihood = 0
        if Q[0][3] != 0:
            actions_likelihood = scipy.stats.norm(Q[0][2], Q[0][3]).pdf(best_actions_sampled_Q_value)
        return [best_action, best_actions_sampled_Q_value, actions_likelihood]
    
    def act_on_with_Q(self, beliefstate):
        '''
        Main policy method: mapping of belief state to system action.
        
        This method is automatically invoked by the agent at each turn after tracking the belief state.
        
        May initially return 'hello()' as hardcoded action. Keeps track of last system action and last belief state.  
        
        :param state: the belief state to act on
        :type state: :class:`~utils.DialogueState.DialogueState`
        :param hyps: n-best-list of semantic interpretations
        :type hyps: list
        :returns: the next system action of type :class:`~utils.DiaAct.DiaAct`
        '''
        if self.lastSystemAction is None and self.startwithhello:
            _systemAct , Q = 'hello()', float("-inf")
        else:
            _systemAct, Q = self.nextAction_with_Q(beliefstate)
        self.lastSystemAction = _systemAct
        self.prevbelief = beliefstate
        
        systemAct = DiaAct.DiaAct(_systemAct)
        return systemAct, Q
    
    def nextAction_with_Q(self, belief):
        '''
        Selects next action to take based on the current belief and a list of non executable actions
        NOT Called by BCM
        
        :param belief:
        :type belief:
        :param hyps:
        :type hyps:
        :returns:
        '''
        nonExecutableActions = self.actions.getNonExecutable(belief, self.lastSystemAction)
        
        currentstate = self.get_State(belief)
        executable = self._createExecutable(nonExecutableActions)
        
#         print "-------------- non-executable actions: {}".format(nonExecutableActions)
#         print "--------------     executable actions: "
#         for act in executable:
#             print act.toString()
#         
#         print "################## GPState"
#         print currentstate._bstate

        if len(executable) < 1:
            return None, float("-inf")

         
        """
        ordered_actions_with_Qsamples = self.learner.policy(state=currentstate, kernel=self.kernel, executable=executable)
        best_action = ordered_actions_with_Qsamples[0][0].act  # [0][1] is sampled Q value
        self.episode.latest_Q_sample_from_choosen_action = ordered_actions_with_Qsamples[0][1]
        """
        best_action, actions_sampledQ, actions_likelihood = self.learner.policy(
                                                                state=currentstate, kernel=self.kernel, executable=executable)
        
        logger.debug('policy activated')
        
        summaryAct = self._actionString(best_action.act)
        
        if self.learning:                    
            best_action.actions_sampled_Qvalue = actions_sampledQ
            best_action.likelihood_choosen_action = actions_likelihood            
            
        self.actToBeRecorded = best_action
        # Finally convert action to MASTER ACTION
        masterAct = self.actions.Convert(belief, summaryAct, self.lastSystemAction)
        return masterAct, actions_sampledQ
    
    def hasExecutableAction(self, belief):
        nonExecutableActions = self.actions.getNonExecutable(belief, self.lastSystemAction)
        return len(nonExecutableActions) != len(self.actions.action_names) 
        
    
class FeudalGPSubPolicyRel(FeudalGPSubPolicy):
    '''
    An implementation of the dialogue policy based on Gaussian process and the GPSarsa algorithm to optimise actions where states are GPState and actions are GPAction.
    
    The class implements the public interfaces from :class:`~Policy.Policy` and :class:`~PolicyCommittee.CommitteeMember`.
    '''
    def __init__(self, domainString, learning, sharedParams=None, rolename = None):
        super(FeudalGPSubPolicyRel, self).__init__(domainString,learning,sharedParams, rolename)
        
        self.actions = SummaryActionRelOnly.SummaryActionRelOnly(domainString, False, self.useconfreq,rolename)
        # Total number of system actions.
        self.numActions = len(self.actions.action_names)
        
    def act_on(self, beliefstate):
        '''
        Main policy method: mapping of belief state to system action.
        
        This method is automatically invoked by the agent at each turn after tracking the belief state.
        
        May initially return 'hello()' as hardcoded action. Keeps track of last system action and last belief state.  
        
        :param state: the belief state to act on
        :type state: :class:`~utils.DialogueState.DialogueState`
        :param hyps: n-best-list of semantic interpretations
        :type hyps: list
        :returns: the next system action of type :class:`~utils.DiaAct.DiaAct`
        '''
        if self.lastSystemAction is None and self.startwithhello:
            _systemAct = 'hello()'
        else:
            _systemAct = self.nextAction(beliefstate)
        self.lastSystemAction = _systemAct
        self.prevbelief = beliefstate
        
        systemAct = DActEntity.DiaActEntity(_systemAct)
        return systemAct
    
    def act_on_with_Q(self, beliefstate):
        '''
        Main policy method: mapping of belief state to system action.
        
        This method is automatically invoked by the agent at each turn after tracking the belief state.
        
        May initially return 'hello()' as hardcoded action. Keeps track of last system action and last belief state.  
        
        :param state: the belief state to act on
        :type state: :class:`~utils.DialogueState.DialogueState`
        :param hyps: n-best-list of semantic interpretations
        :type hyps: list
        :returns: the next system action of type :class:`~utils.DiaAct.DiaAct`
        '''
        if self.lastSystemAction is None and self.startwithhello:
            _systemAct , Q = 'hello()', float("-inf")
        else:
            _systemAct, Q = self.nextAction_with_Q(beliefstate)
        self.lastSystemAction = _systemAct
        self.prevbelief = beliefstate
        
        systemAct = DActEntity.DiaActEntity(_systemAct)
        return systemAct, Q
    
    def get_State(self, beliefstate, keep_none=False):     
        '''
        Called by BCM
        
        :param beliefstate:
        :type beliefstate:
        :param keep_none:
        :type keep_none:
        '''
        return GPStateRel(beliefstate, keep_none=keep_none, replace=self.replace, domainString=self.domainString)
    
    def constructInitialBelief(self, features):
        belief = {'beliefs': dict(), 'features': deepcopy(features)}
        
        rel = self.rolename.split('#')
        common_slots = Ontology.global_ontology.get_common_slots(rel[0],rel[1])
        # {u'area#area': 0.0, u'pricerange#pricerange': 0.0}
        belief['beliefs']['requested'] = dict()
        for slot in common_slots:
            s = slot
            belief['beliefs'][s] = {'**NONE**': 1.0, 'dontcare' : 0.0, '=': 0.0}
            belief['beliefs']['requested'][s] = 0.0
            
        return belief
    
class FeudalGPSubPolicyObj(FeudalGPSubPolicy):
    '''
    An implementation of the dialogue policy based on Gaussian process and the GPSarsa algorithm to optimise actions where states are GPState and actions are GPAction.
    
    The class implements the public interfaces from :class:`~Policy.Policy` and :class:`~PolicyCommittee.CommitteeMember`.
    '''
    def __init__(self, domainString, learning, sharedParams=None, rolename = None):
        super(FeudalGPSubPolicyRel, self).__init__(domainString,learning,sharedParams, rolename)
        
        self.actions = SummaryAction.SummaryAction(domainString, False, self.useconfreq)
        
        # Total number of system actions.
        self.numActions = len(self.actions.action_names)
        
    def get_State(self, beliefstate, keep_none=False):     
        '''
        Called by BCM
        
        :param beliefstate:
        :type beliefstate:
        :param keep_none:
        :type keep_none:
        '''
        return GPStateObj(beliefstate, keep_none=keep_none, replace=self.replace, domainString=self.domainString)
        
    def act_on(self, beliefstate):
        '''
        Main policy method: mapping of belief state to system action.
        
        This method is automatically invoked by the agent at each turn after tracking the belief state.
        
        May initially return 'hello()' as hardcoded action. Keeps track of last system action and last belief state.  
        
        :param state: the belief state to act on
        :type state: :class:`~utils.DialogueState.DialogueState`
        :param hyps: n-best-list of semantic interpretations
        :type hyps: list
        :returns: the next system action of type :class:`~utils.DiaAct.DiaAct`
        '''
        if self.lastSystemAction is None and self.startwithhello:
            _systemAct = 'hello()'
        else:
            _systemAct = self.nextAction(beliefstate)
        self.lastSystemAction = _systemAct
        self.prevbelief = beliefstate
        
        systemAct = DActEntity.DiaActEntity(_systemAct)
        return systemAct
    
class GPStateObj(policy.GPPolicy.GPState):
    '''
    Currently an exact copy of the original GPState
    '''
    pass

class GPStateRel(policy.GPPolicy.GPState):  
    def extractSimpleBelief(self, b, replace={}):
        '''
        From the belief state b extracts discourseAct, method, requested slots, name, goal for each slot,
        history whether the offer happened, whether last action was inform none, and history features.
        Sets self._bstate
        '''
        with_other = 0
        without_other = 0
        self.isFullBelief = True
        
        for elem in b['beliefs'].keys():
            if elem == 'discourseAct':
                self._bstate["goal_discourseAct"] = b['beliefs'][elem].values()
                without_other +=1
            elif elem == 'method':
                self._bstate["goal_method"] = b['beliefs'][elem].values()
                without_other +=1
            elif elem == 'requested' :
                for slot in b['beliefs'][elem]:
                    cur_slot=slot
                    if len(replace) > 0:
                        cur_slot = replace[cur_slot]
                    self._bstate['hist_'+cur_slot] = self.extractSingleValue(b['beliefs']['requested'][slot])
                    without_other +=1
            else:
                if elem == 'name':
                    self._bstate[elem] = self.extractBeliefWithOther(b['beliefs']['name'])
                    with_other +=1
                else:
                    cur_slot=elem
                    if len(replace) > 0:
                        cur_slot = replace[elem]

                    self._bstate['goal_'+cur_slot] = self.extractBeliefWithOther(b['beliefs'][elem])
                    with_other += 1

#                     additionalSlots = 2
                    # if elem not in Ontology.global_ontology.get_system_requestable_slots(self.domainString):
                    #     additionalSlots = 1
#                     if len(self._bstate['goal_'+cur_slot]) !=\
#                          Ontology.global_ontology.get_len_informable_slot(self.domainString, slot=elem)+additionalSlots:
#                         print self._bstate['goal_'+cur_slot]
#                         logger.error("Different number of values for slot "+cur_slot+" "+str(len(self._bstate['goal_'+cur_slot]))+\
#                             " in ontology "+ str(Ontology.global_ontology.get_len_informable_slot(self.domainString, slot=elem)+2)) 
                    

#         self._bstate["hist_offerHappened"] = self.extractSingleValue(1.0 if b['features']['offerHappened'] else 0.0)
#         without_other +=1
#         self._bstate["hist_lastActionInformNone"] = self.extractSingleValue(
#                                                                 1.0 if len(b['features']['informedVenueSinceNone'])>0 else 0.0)
        without_other +=1
        for i,inform_elem in enumerate(b['features']['inform_info']):
            self._bstate["hist_info_"+str(i)] = self.extractSingleValue(1.0 if inform_elem else 0.0)
            without_other +=1
            
        # Tom's speedup: convert belief dict to numpy vector 
        self.beliefStateVec = self.slowToFastBelief(self._bstate)

        return
