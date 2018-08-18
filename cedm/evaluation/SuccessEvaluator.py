###############################################################################
# PyDial: Multi-domain Statistical Spoken Dialogue System Software
###############################################################################
#
# Copyright 2015-16  Cambridge University Engineering Department 
# Dialogue Systems Group
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
SuccessEvaluator.py - module for determining objective and subjective dialogue success 
======================================================================================

Copyright CUED Dialogue Systems Group 2016

.. seealso:: PyDial Imports/Dependencies: 

    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger` |.|
    import :mod:`utils.DiaAct` |.|
    import :mod:`ontology.Ontology` |.|
    import :class:`evaluation.EvaluationManager.Evaluator` |.|

************************

'''
__author__ = "cued_dialogue_systems_group"

import evaluation.SuccessEvaluator
from utils import ContextLogger
from ontology import Ontology
import numpy as np

logger = ContextLogger.getLogger('')

class ObjectiveERSuccessEvaluator(evaluation.SuccessEvaluator.ObjectiveSuccessEvaluator):
    '''
    This class provides a reward model based on objective success. For simulated dialogues, the goal of the user simulator is compared with the the information the system has provided. 
    For dialogues with a task file, the task is compared to the information the system has provided. 
    '''
    
    def __init__(self, domainString):
        super(ObjectiveERSuccessEvaluator, self).__init__(domainString)
        
        # only for nice prints
        self.evaluator_label = "objective ER success evaluator"
               
    def _getFinalReward(self,finalInfo):
        '''
        Computes the final reward using finalInfo. Should be overridden by sub-class if values others than 0 should be returned.
        
        :param finalInfo: parameters necessary for computing the final reward, eg., task description or subjective feedback.
        :type finalInfo: dict
        :return: int -- the final reward, default 0.
        '''
        if finalInfo is not None and isinstance(finalInfo, dict):
            if 'usermodel' in finalInfo: # from user simulator
                um = finalInfo['usermodel']
                if um is None:
                    self.outcome = False
                elif self.domainString not in um:
                    self.outcome = False
                else:
                    requests = um[self.domainString].goal.requests
                    '''if self.last_venue_recomended is None:
                        logger.dial('Fail :( User requests: {}, Venue recomended: {}'.format(requests, self.venue_recommended))
                    else:
                        if self.venue_recommended and None not in requests.values():
                            self.outcome = True
                            logger.dial('Success! User requests: {}, Venue recomended: {}'.format(requests, self.venue_recommended))
                        else:
                            logger.dial('Fail :( User requests: {}, Venue recomended: {}'.format(requests, self.venue_recommended))'''
                    if None not in requests.values():
                        valid_venue = self._isValidVenue(requests['name'], self.user_goal)
                        if valid_venue:
                            self.outcome = True
                            logger.dial(
                                'Success! User requests: {}'.format(requests))
                        else:
                            logger.dial(
                                'Fail :( User requests: {}'.format(requests))
                    else:
                        logger.dial(
                            'Fail :( User requests: {}'.format(requests))
            elif 'task' in finalInfo: # dialogue server with tasks
                task = finalInfo['task']
                if self.DM_history is not None:
                    informs = self._get_informs_against_each_entity()
                    if informs is not None:
                        for ent in informs.keys():
                            if task is None:
                                self.outcome = True   # since there are no goals, lets go with this ... 
                            elif self.domainString not in task:
                                logger.warning("This task doesn't contain the domain: %s" % self.domainString)
                                logger.debug("task was: " + str(task))  # note the way tasks currently are, we dont have 
                                # the task_id at this point ...
                                self.outcome = True   # This is arbitary, since there are no goals ... lets say true?
                            elif ent in str(task[self.domainString]["Ents"]):
                                # compare what was informed() against what was required by task:
                                required = str(task[self.domainString]["Reqs"]).split(",")
                                self.outcome = True
                                for req in required:
                                    if req == 'name':
                                        continue
                                    if req not in ','.join(informs[ent]): 
                                        self.outcome = False

        return self.outcome * self.successReward - (not self.outcome) * self.failPenalty
    
    def _get_informs_against_each_entity(self):
        if len(self.DM_history) == 0:
            return None
        informs = {}
        currentEnt = None
        for act in self.DM_history:
            if 'inform(' in act:
                details = act.split("(")[1].split(",")
                details[-1] = details[-1][0:-1]  # remove the closing )
                if not len(details):
                    continue
                if "name=" in act:
                    for detail in details:
                        if "name=" in detail:
                            currentEnt = detail.split("=")[1].strip('"')
                            details.remove(detail)
                            break  # assumes only 1 name= in act -- seems solid assumption
                    
                    if currentEnt in informs.keys():
                        informs[currentEnt] += details
                    else:
                        informs[currentEnt] = details
                elif currentEnt is None:
                    logger.warning("Shouldn't be possible to first encounter an inform() act without a name in it")
                else:
                    logger.warning('assuming inform() that does not mention a name refers to last entity mentioned')
                    informs[currentEnt] += details
        return informs

    
    
    def _update_mentioned_value(self, act):
        # internal, called by :func:`RewardComputer.get_reward` for both sys and user acts to update values mentioned in dialog
        #
        # :param act: sys or user dialog act
        # :type act: :class:`DiaAct.DiaAct`
        # :return: None
        
        sys_requestable_slots = Ontology.global_ontology.get_system_requestable_slots(self.domainString)
        for item in act.items:
            if item.slot in sys_requestable_slots and item.val not in [None, '**NONE**', 'none']:
                self.mentioned_values[item.slot].add(item.val)
                
                
    def _getResultString(self, outcomes):
        num_dialogs = len(outcomes)
        from scipy import stats
        if num_dialogs < 2:
            tinv = 1
        else:
            tinv = stats.t.ppf(1 - 0.025, num_dialogs - 1)
        return 'Average success = {0:0.2f} +- {1:0.2f}'.format(100 * np.mean(outcomes), \
                                                            100 * tinv * np.std(outcomes) / np.sqrt(num_dialogs))
                
    
#END OF FILE
