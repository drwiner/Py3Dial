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
EvaluationManager.py - module for determining the reward
==========================================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`ontology.OntologyUtils` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger` 

************************

'''

__author__ = "cued_dialogue_systems_group"
from evaluation import EvaluationManager as em
from utils import ContextLogger
logger = ContextLogger.getLogger('')


    
class EvaluationManager(em.EvaluationManager):
    '''
    The evaluation manager manages the evaluators for all domains. It supports two types of reward: a turn-level reward and a dialogue-level reward. 
    The former is accessed using :func:`turnReward` and the latter using :func:`finalReward`.
    You can either use one or both methods for reward computing.
    
    An example where both are used in the traditional reward computation where each turn is penalised with a small negative reward (which is realised with :func:`turnReward`)
    and in the end, the dialogue is rewarded with a big positive reward given the overall dialogue (which is realised with :func:`finalReward`).
    '''
    
    def turnReward(self, domainString, turnInfo):
        '''
        Computes the turn reward for the given domain using turnInfo by delegating to the domain evaluator.
        
        :param domainString: the domain string unique identifier.
        :type domainString: str
        :param turnInfo: parameters necessary for computing the turn reward, eg., system act or model of the simulated user.
        :type turnInfo: dict
        :return: int -- the turn reward for the given domain.
        '''
        # Replaces: for sim: reward_and_success, for texthub/dialoguserver: per_turn_reward, add_DM_history
        # turnInfo: {simulatedUserModel, sys_act} 
        
        dString = turnInfo['state'].entityFocus
        if domainString == dString:
            return 0
        
        if dString in self.SPECIAL_DOMAINS:
            return 0
        
        if self.domainEvaluators[dString] is None:
            self._bootup_domain(dString)
            
        turnInfo['belief'] = turnInfo['state'].getFocusBelief()
        
        return self.domainEvaluators[dString].turnReward(turnInfo)
    
#     def finalReward(self, domainString, finalInfo):
#         '''
#         Computes the final reward for the given domain using finalInfo by delegating to the domain evaluator.
#         
#         :param domainString: the domain string unique identifier.
#         :type domainString: str
#         :param finalInfo: parameters necessary for computing the final reward, eg., task description or subjective feedback.
#         :type finalInfo: dict
#         :return: int -- the final reward for the given domain.
#         '''
#         # Replaces: for all: finalise_dialogue, for dialogue_server/texthub: objective_success_by_task
#         # finalInfo: {task, subjectiveFeedback}
#         
#         if domainString in self.SPECIAL_DOMAINS:
#             return 0
#         
#         self.final_reward[domainString] = self.domainEvaluators[domainString].finalReward(finalInfo)
#         return self.final_reward[domainString]
#     
#     def finalRewards(self, finalInfo=None):
#         '''
#         Computes the :func:`finalReward` method for all domains where it has not been computed yet.
#         
#         :param finalInfo: parameters necessary for computing the final rewards, eg., task description or subjective feedback. Default is None
#         :type finalInfo: dict
#         :returns: dict -- mapping of domain to final rewards 
#         '''
#         for domain in self.final_reward:
#             if self.final_reward[domain] is None and self.domainEvaluators[domain] is not None:
#                 self.finalReward(domain, finalInfo)
#                 
#         return self.final_reward
    
    

# END OF FILE
