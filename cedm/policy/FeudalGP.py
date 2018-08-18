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
GPPolicy.py - Gaussian Process policy 
============================================

Copyright CUED Dialogue Systems Group 2015 - 2017

   
**Relevant Config variables** [Default values]::

    [gppolicy]
    kernel = polysort
    thetafile = ''    

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`policy.GPLib` |.|
    import :mod:`policy.Policy` |.|
    import :mod:`policy.PolicyCommittee` |.|
    import :mod:`ontology.Ontology` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger`

************************

'''

__author__ = "cued_dialogue_systems_group"



from GPPolicy import GPPolicy
from policy.GPPolicy import Kernel, GPState
from policy.GPLib import GPSARSA
from ontology import Ontology
import SummaryActionMaster
from FeudalSubPolicy import FeudalGPSubPolicy,FeudalGPSubPolicyRel
from policy import SummaryUtils
from cedm.utils import DActEntity
from utils import ContextLogger, Settings
import itertools
logger = ContextLogger.getLogger('')

class FeudalGP(GPPolicy):
    '''
    An implementation of the dialogue policy based on Gaussian process and the GPSarsa algorithm to optimise actions where states are GPState and actions are GPAction.
    
    The class implements the public interfaces from :class:`~Policy.Policy` and :class:`~PolicyCommittee.CommitteeMember`.
    '''
    def __init__(self, domainString, learning, sharedParams=None, sharedPolicies = {}):
        super(FeudalGP, self).__init__(domainString,learning,sharedParams)
        
        self.winnerPolicy = None
        
        self.policies = {}
        self.policyPartOfDialogue = {self.domainString : False} # entity is always part of dialogue, unsure about relations
        
        self.useMasterPolicy = False
        if Settings.config.has_option('policy_'+domainString, 'usemaster'):
            self.useMasterPolicy = Settings.config.getboolean('policy_'+domainString, 'usemaster')
            
        self.sharedPolicies = {}
        if Settings.config.has_option('policy_'+domainString, 'sharedpolicies'):
            useSharedPolicies = Settings.config.getboolean('policy_'+domainString, 'sharedpolicies')
            if useSharedPolicies:
                self.sharedPolicies = sharedPolicies
        
        if self.useMasterPolicy:
            self.actions = SummaryActionMaster.SummaryActionMaster(domainString, False, self.useconfreq)
            # Total number of system actions.
            self.numActions = len(self.actions.action_names)
            
            
            self.kernel = Kernel(self.kerneltype, self.theta, None, self.action_kernel_type, self.actions.action_names, domainString)
            
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
            
            outpolicyfile += "_{}".format('master')
            inpolicyfile += "_{}".format('master')
            
            self.learner = GPSARSA(inpolicyfile,outpolicyfile,domainString=domainString, learning=self.learning, sharedParams=sharedParams)
        
        
        
    def act_on(self, state):
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
        # first get belief of entity in focus
#         entity_beliefstate = state.getEntity(state.entityFocus).belief
        entity_beliefstate = state.getMergedBelief(state.entityFocus,newOne = True, getConflict = False)
        master_beliefstate = state.getMergedBelief(state.entityFocus,newOne = True, getConflict = True)
#         teststate = state.getMergedBelief(state.entityFocus,True)
#         
#         if entity_beliefstate != teststate:
#             for slot in teststate['beliefs']:
#                 if teststate['beliefs'][slot] != entity_beliefstate['beliefs'][slot]:
#                     logger.error('States differ. old: {} new: {}'.format(entity_beliefstate['beliefs'][slot],teststate['beliefs'][slot]))
        # then update db features using merging rule
        entity_beliefstate['features']['inform_info'] = self._updateDBfeatures(state.getMergedBelief(state.entityFocus,False),state.entityFocus)
        master_beliefstate['features']['inform_info'] = self._updateDBfeatures(state.getMergedBelief(state.entityFocus,False),state.entityFocus)
        
        # now get belief of relation
        relation_beliefs = state.getExtendedRelationBeliefs(state.entityFocus)
        for rel in relation_beliefs:
            if 'features' not in relation_beliefs[rel]:
                relation_beliefs[rel]['features'] = dict()
            relation_beliefs[rel]['features']['inform_info'] = entity_beliefstate['features']['inform_info']
        
        if self.lastSystemAction is None and self.startwithhello:
            _systemAct = 'hello()'
        else:
            _systemAct = self.nextAction(entity_beliefstate,relation_beliefs,state.entityFocus,master_beliefstate)
        
        self.lastSystemAction = _systemAct
        
        # not needed as all sub-policies should record their own stuff
#         self.prevbelief = beliefstate
        
        systemAct = DActEntity.DiaActEntity(_systemAct, state.entityFocus)
        return systemAct
    
    def nextAction(self, entity_beliefstate,relation_beliefs,entityFocus,master_beliefstate = None):
        if not self.useMasterPolicy:
            return self.nextActionMaxQ(entity_beliefstate,relation_beliefs,entityFocus)
        else:
            return self.nextActionMaster(entity_beliefstate,relation_beliefs,master_beliefstate,entityFocus)
        
    def nextActionMaster(self, entity_beliefstate, relation_beliefs,master_beliefstate,  entityFocus):
        
        currentstate = self.get_State(master_beliefstate)
        
        relAction = False
        best_rel_act, best_rel_Q, best_rel_policy = None, float('-inf'), None
        for rel in relation_beliefs:
            self.policyPartOfDialogue[rel] = False
            if rel not in self.policies:
                self.policyPartOfDialogue[rel] = True
                if rel not in self.sharedPolicies:
                    self.sharedPolicies[rel] = FeudalGPSubPolicyRel(entityFocus,self.learning,rolename=rel)
                self.policies[rel] = self.sharedPolicies[rel]
                            
            rel_act, relQ = self.policies[rel].act_on_with_Q(relation_beliefs[rel])
            if relQ > best_rel_Q:
                best_rel_Q = relQ
                best_rel_act = rel_act
                best_rel_policy = rel
            
            if rel in self.policies:
                if self.policies[rel].hasExecutableAction(relation_beliefs[rel]):
                    relAction = True
                    
        if entityFocus not in self.policies:
            self.policies[entityFocus] = FeudalGPSubPolicy(entityFocus,self.learning,rolename=entityFocus)
            
        # best_act, bestQ
        best_act, _ = self.policies[entityFocus].act_on_with_Q(entity_beliefstate)
                    
        nonexecutables = ['rel'] if not relAction else []
        
        executables = self._createExecutable(nonexecutables)
        # best_action, actions_sampledQ, actions_likelihood 
        best_action, _, _ = self.learner.policy(
                                                                state=currentstate, kernel=self.kernel, executable=executables)
        self.actToBeRecorded = best_action
        self.prevbelief = currentstate
        
        masteract = None
        if best_action.act == 'rel':
            self.winnerPolicy = best_rel_policy
            masteract = best_rel_act 
        elif best_action.act == 'obj':
            self.winnerPolicy = entityFocus
            masteract = best_act
        else:
            logger.error('Master action unknown. {}'.format(best_action))
            
        return masteract
    
    def nextActionMaxQ(self, entity_beliefstate, relation_beliefs, entityFocus):
        # check if policies loaded
        if entityFocus not in self.policies:
            self.policies[entityFocus] = FeudalGPSubPolicy(entityFocus,self.learning,rolename=entityFocus)
            
        for rel in relation_beliefs:
            self.policyPartOfDialogue[rel] = False
            if rel not in self.policies:
                self.policyPartOfDialogue[rel] = True
                self.policies[rel] = FeudalGPSubPolicyRel(entityFocus,self.learning,rolename=rel)
        
        
        
        
        best_act, bestQ = self.policies[entityFocus].act_on_with_Q(entity_beliefstate)
        self.winnerPolicy = entityFocus
              
        for rel in relation_beliefs:
            rel_act, relQ = self.policies[rel].act_on_with_Q(relation_beliefs[rel])
            if relQ > bestQ:
                bestQ = relQ
                best_act = rel_act
                self.winnerPolicy = rel
                
        return best_act
    
    def record(self, reward, domainInControl = None, weight = None, state=None, action=None):
        '''
        Records the current turn reward along with the last system action and belief state.
        
        This method is automatically executed by the agent at the end of each turn.
        
        To change the type of state/action override :func:`~convertStateAction`. By default, the last master action is recorded. 
        If you want to have another action being recorded, eg., summary action, assign the respective object to self.actToBeRecorded in a derived class. 
        
        :param reward: the turn reward to be recorded
        :type reward: int
        :param domainInControl: the domain string unique identifier of the domain the reward originates in
        :type domainInControl: str
        :param weight: used by committee: the weight of the reward in case of multiagent learning
        :type weight: float
        :param state: used by committee: the belief state to be recorded
        :type state: dict
        :param action: used by committee: the action to be recorded
        :type action: str
        :returns: None
        '''
        
        for policy in self.policyPartOfDialogue:
            if self.policyPartOfDialogue[policy]:
                # need to fill episode with pass actions and initial states until current turn
                e = self.policies[self.domainString].episodes[self.domainString]
                features = {'inform_info' : self.policies[self.domainString].prevbelief['features']['inform_info']}
                state = self.policies[policy].constructInitialBelief(features)
                for rew, state in itertools.izip(e.rtrace, e.strace):
                    features = {'inform_info' : self._constructInformInfoFromGPState(state)}
                    state = self.policies[policy].constructInitialBelief(features)
                    self.policies[policy].record(reward = rew, domainInControl = self.domainString, state = state, action = 'pass')
#                 for rew in e.rtrace:
#                     self.policies[policy].record(reward = rew, domainInControl = self.domainString, state = state, action = 'pass')
        
        
        if self.winnerPolicy is not None:
            self.policies[self.winnerPolicy].record(reward,domainInControl = self.domainString)
            for policy in self.policies:
                if policy == self.winnerPolicy:
                    continue
                if policy not in self.policyPartOfDialogue: # happens only for relations as they are dynamic
                    features = {'inform_info' : self.policies[self.winnerPolicy].prevbelief['features']['inform_info']}
                    state = self.policies[policy].constructInitialBelief(features)
                    self.policies[policy].record(reward,action="pass", domainInControl = self.domainString, state = state)
                else:
                    self.policies[policy].record(reward,action="pass", domainInControl = self.domainString)
        else:
            logger.error('recording for feudal policy but no action has been taken yet.')    
        
        if self.useMasterPolicy:
            super(FeudalGP, self).record(reward)
    
    def finalizeRecord(self, reward, domainInControl = None):
        '''
        Records the final reward along with the terminal system action and terminal state. To change the type of state/action override :func:`~convertStateAction`.
        
        This method is automatically executed by the agent at the end of each dialogue.
        
        :param reward: the final reward
        :type reward: int
        :param domainInControl: used by committee: the unique identifier domain string of the domain this dialogue originates in, optional
        :type domainInControl: str
        :returns: None
        '''
        for policy in self.policies:
            self.policies[policy].finalizeRecord(reward,domainInControl) 
            
            
        if self.useMasterPolicy:
            super(FeudalGP, self).finalizeRecord(reward, domainInControl)
        return
    
    def train(self):
        for policy in self.policies:
            self.policies[policy].train()
            
        if self.useMasterPolicy:
            super(FeudalGP, self).train()
        return
    
    def savePolicy(self, FORCE_SAVE=False):
        for policy in self.policies:
            self.policies[policy].savePolicy(FORCE_SAVE)
        if self.useMasterPolicy:
            super(FeudalGP, self).savePolicy(FORCE_SAVE)
        return
    
    def restart(self):
        super(FeudalGP, self).restart()
        self.winnerPolicy = None
        self.policyPartOfDialogue = {self.domainString: False} # entity is always part of dialogue, unsure about relations
        
        for policy in self.policies:
            self.policies[policy].restart()
        return
    
    def get_State(self, beliefstate, keep_none=False):     
        '''
        Called by BCM
        
        :param beliefstate:
        :type beliefstate:
        :param keep_none:
        :type keep_none:
        '''
        return GPStateMaster(beliefstate, keep_none=keep_none, replace=self.replace, domainString=self.domainString)
    
    def _updateDBfeatures(self,belief,eType):
        features = []
        for numAccepted in range(1,6):
            temp =  SummaryUtils.actionSpecificInformSummary(belief, numAccepted, eType)
            features += temp
        return features
    
    def _constructInformInfoFromGPState(self, state):
#         state._bstate['hist_info_0']
        inform_info_dict = {}
        for s in state._bstate:
            if 'hist_info' in s:
                no = int(s.split('_')[2])
                inform_info_dict[no] = state._bstate[s][0] == 1.0
        
        sortedKeys = sorted(inform_info_dict.keys())
        
        inform_info = []
        for key in sortedKeys:
            inform_info.append(inform_info_dict[key])
        return inform_info
    
class GPStateMaster(GPState):
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

                    additionalSlots = 2
                    # if elem not in Ontology.global_ontology.get_system_requestable_slots(self.domainString):
                    #     additionalSlots = 1
                    if len(self._bstate['goal_'+cur_slot]) !=\
                         Ontology.global_ontology.get_len_informable_slot(self.domainString, slot=elem)+additionalSlots:
                        print self._bstate['goal_'+cur_slot]
                        logger.error("Different number of values for slot "+cur_slot+" "+str(len(self._bstate['goal_'+cur_slot]))+\
                            " in ontology "+ str(Ontology.global_ontology.get_len_informable_slot(self.domainString, slot=elem)+2)) 
                    

        self._bstate["hist_offerHappened"] = self.extractSingleValue(1.0 if b['features']['offerHappened'] else 0.0)
        without_other +=1
        self._bstate["hist_lastActionInformNone"] = self.extractSingleValue(
                                                                1.0 if len(b['features']['informedVenueSinceNone'])>0 else 0.0)
        without_other +=1
        for i,inform_elem in enumerate(b['features']['inform_info']):
            self._bstate["hist_info_"+str(i)] = self.extractSingleValue(1.0 if inform_elem else 0.0)
            without_other +=1
            
        if 'relationConflict' in b['features']:
            self._bstate['hist_conflict'] = self.extractSingleValue(1.0 if b['features']['relationConflict'] else 0.0)
            
        # Tom's speedup: convert belief dict to numpy vector 
        self.beliefStateVec = self.slowToFastBelief(self._bstate)

        return
    
        
# END OF FILE