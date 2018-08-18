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
SummaryActionRel.py - Mapping between summary and master actions
=============================================================

Copyright CUED Dialogue Systems Group 2015 - 2017, 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`policy.SummaryUtils` |.|
    import :mod:`ontology.Ontology` |.|
    import :mod:`utils.ContextLogger` |.|
    import :mod:`utils.Settings`

************************

'''


__author__ = "cued_dialogue_systems_group"

import policy.SummaryAction
import policy.SummaryUtils as SummaryUtils
import SummaryUtilsRel
from utils import ContextLogger
from ontology import Ontology
logger = ContextLogger.getLogger('')

MAX_NUM_ACCEPTED = 10


class SummaryActionRel(policy.SummaryAction.SummaryAction):
    '''
    The summary action class encapsulates the functionality of a summary action along with the conversion from summary to master actions.
    
    .. Note::
        The list of all possible summary actions are defined in this class.
    '''
    def __init__(self, domainString, empty=False, confreq=False, relationOnly=False):
        '''
        Records what domain the class is instantiated for, and what actions are available

        :param domainString: domain tag
        :type domainString: string
        :param empty: None
        :type empty: bool
        :param confreq: representing if the action confreq is used
        :type confreq: bool
        '''
        
        super(SummaryActionRel, self).__init__(domainString, empty, confreq)
        
        self._array_slot_summary_rel = None
        
        

        if not empty:
            relations = Ontology.global_ontology.get_common_slots_for_all_types(domainString)
            for domain in relations:
                for slot in relations[domain]:
                    self.action_names.append("request_rel_" + domain + "_" + slot)
                    self.action_names.append("confirm_rel_" + domain + "_" + slot)
                    self.action_names.append("select_rel_" + domain + "_" + slot)
        self.reset()

    def reset(self):
        self.alternatives_requested = False

    def Convert(self, belief, action, lastSystemAction):
        '''
        Converts the given summary action into a master action based on the current belief and the last system action.

        :param belief: the current master belief
        :type belief: dict
        :param action: the summary action to be converted to master action
        :type action: string
        :param lastSystemAction: the system action of the previous turn
        :type lastSystemAction: string
        '''
        if 'rel' not in action:
            output = super(SummaryActionRel, self).Convert(belief, action, lastSystemAction)
        else:
            self._array_slot_summary_rel = SummaryUtilsRel.arraySlotSummaryRel(belief, self.domainString)
            output = None
            logger.dial('system summary act: {}.'.format(action))

            if "request_" in action:
                output = self.getRequestRel(action.split("_")[2:])
            elif "select_" in action:
                output = self.getSelectRel(action.split("_")[2:])
            elif "confirm_" in action:
                output = self.getConfirmRel(action.split("_")[2:])
            else:
                output = ""
                logger.error("Unknown action: " + action)
        return output

    # MASK OVER SUMMARY ACTION SET
    # ------------------------------------------------------------------------------------

    def getNonExecutable(self, belief, lastSystemAction):
        '''
        Set of rules defining the mask over the action set, given the current belief state
        :param belief: the current master belief
        :type belief: dict
        :param lastSystemAction: the system action of the previous turn
        :type lastSystemAction: string
        :return: list of non-executable (masked) actions
        '''

        array_slot_summary = SummaryUtils.arraySlotSummary(belief, self.domainString)
        array_slot_summary_rel = SummaryUtilsRel.arraySlotSummaryRel(belief, self.domainString)
        global_summary = SummaryUtils.globalSummary(belief, self.domainString)
        if global_summary['GLOBAL_BYALTERNATIVES'] and not global_summary['GLOBAL_THANKYOU'] and not global_summary['GLOBAL_ACK']:
            self.alternatives_requested = True

        nonexec = ['pass']

        for action in self.action_names:
            mask_action = False
            
            if 'rel' in action:
                if "request_" in action:
                    nonexec.append(action)
#                     pass
#                     if mask_action and self.request_mask:
#                         nonexec.append(action)
    
                elif "select_" in action:
                    slot_summary = array_slot_summary_rel['_'.join(action.split('_')[2:])]
                    top_prob = slot_summary['TOPHYPS'][0][1]
                    sec_prob = slot_summary['TOPHYPS'][1][1]
                    if top_prob == 0 or sec_prob == 0:
                        mask_action = True
                    if mask_action and self.request_mask:
                        nonexec.append(action)
    
                elif "confirm_" in action:
                    slot_summary = array_slot_summary_rel['_'.join(action.split('_')[2:])]
                    top_prob = slot_summary['TOPHYPS'][0][1]
                    if top_prob == 0:
                        mask_action = True
                    if mask_action and self.request_mask:
                        nonexec.append(action)
    
                
            else:
                if action == "inform":
                    acceptance_list = SummaryUtils.getTopBeliefs(belief, domainString=self.domainString)
                    discriminable = SummaryUtils.acceptanceListCanBeDiscriminated(acceptance_list,
                                                                                                         self.domainString)
                    if not global_summary['GLOBAL_BYCONSTRAINTS']:
                        mask_action = True
                    if global_summary['GLOBAL_COUNTACCEPTED'] < self.inform_count_accepted and discriminable:
                        mask_action = True
                    if mask_action and self.inform_mask:
                        nonexec.append(action)
    
                elif action == "inform_byname":
                    if not global_summary['GLOBAL_BYNAME']:
                        mask_action = True
                    if belief['features']['lastInformedVenue'] == '' \
                            and SummaryUtils.getTopBelief(belief['beliefs']['name'])[0] == '**NONE**' :
                        mask_action = True
                    if mask_action and self.inform_mask:
                        nonexec.append(action)
    
                elif action == "inform_alternatives":
                    if not self.alternatives_requested:
                        mask_action = True
                    if belief['features']['lastInformedVenue'] == '':
                        mask_action = True
                    if mask_action and self.inform_mask:
                        nonexec.append(action)
    
                elif action == "bye":
                    if not global_summary['GLOBAL_FINISHED']:
                        mask_action = True
                    if mask_action and self.bye_mask:
                        nonexec.append(action)
    
                elif action == "repeat":
                    if not global_summary['GLOBAL_REPEAT'] or lastSystemAction is None:
                        mask_action = True
                    mask_action = True  # ic340: this action is "deactivated" because simuser doesnt know how to react to it
                    if mask_action:
                        nonexec.append(action)
    
                elif action == "reqmore":
                    if belief['features']['lastInformedVenue'] == '':
                        mask_action = True
                    if mask_action and self.request_mask:
                        nonexec.append(action)
    
                elif action == "restart":
                    if not global_summary['GLOBAL_RESTART']:
                        mask_action = True
                    mask_action = True  # ic340: this action is "deactivated" because simuser doesnt know how to react to it
                    if mask_action:
                        nonexec.append(action)
    
                elif "request_" in action:
                    pass
                    if mask_action and self.request_mask:
                        nonexec.append(action)
    
                elif "select_" in action:
                    slot_summary = array_slot_summary[action.split("_")[1]]
                    top_prob = slot_summary['TOPHYPS'][0][1]
                    sec_prob = slot_summary['TOPHYPS'][1][1]
                    if top_prob == 0 or sec_prob == 0:
                        mask_action = True
                    if mask_action and self.request_mask:
                        nonexec.append(action)
    
                elif "confirm_" in action:
                    slot_summary = array_slot_summary[action.split("_")[1]]
                    top_prob = slot_summary['TOPHYPS'][0][1]
                    if top_prob == 0:
                        mask_action = True
                    if mask_action and self.request_mask:
                        nonexec.append(action)
    
                elif "confreq_" in action:
                    slot_summary = array_slot_summary[action.split("_")[1]]
                    top_prob = slot_summary['TOPHYPS'][0][1]
                    if top_prob == 0:
                        mask_action = True
                    if mask_action and self.request_mask:
                        nonexec.append(action)

        logger.dial('masked inform actions:' + str([act for act in nonexec if 'inform' in act]))
        return nonexec

    # CONVERTING METHODS FOR EACH SPECIFIC ACT:
    #------------------------------------------------------------------------------------
    
    def getRequestRel(self, slot):
        if len(slot) < 2:
            logger.error('There is no domain + slot information present. Why?')
        s = slot[1].split('#')
        return 'request({}={}#{})'.format(s[0],slot[0],s[1])

    def getConfirmRel(self, slot):
        if len(slot) < 2:
            logger.error('There is no domain + slot information present. Why?')
        s = slot[1].split('#')
        summary = self._array_slot_summary_rel['_'.join(slot)]
        top_value = summary['TOPHYPS'][0][0]
        return 'confirm({}#{}{}{}#{})'.format(self.domainString,s[0], top_value,slot[0],s[1])
    
    def getSelectRel(self, slot):
        summary = self._array_slot_summary[slot]
        top_value = summary['TOPHYPS'][0][0]
        sec_value = summary['TOPHYPS'][1][0]
        return 'select({}="{}",{}="{}")'.format(slot, top_value, slot, sec_value)
    
    # needs to be overridden as requested_slots may contain a mixture of rel and non-rel slots
    def getInformByName(self, belief):
        requested_slots = SummaryUtilsRel.getRequestedSlots(belief)
        name = SummaryUtils.getTopBelief(belief['beliefs']['name'])[0]
        if name == '**NONE**':
            name = belief['features']['lastInformedVenue']
        return SummaryUtilsRel.getInformRequestedSlots(requested_slots, name, self.domainString)

#END OF FILE
