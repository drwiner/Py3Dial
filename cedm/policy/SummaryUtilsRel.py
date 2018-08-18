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
SummaryUtilsRel.py - summarises dialog events for mapping from master to summary belief 
======================================================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

**Basic Usage**: 
    >>> import SummaryUtils
   
.. Note::
        No classes; collection of utility methods

Local module variables::

    global_summary_features:    (list) global actions/methods
    REQUESTING_THRESHOLD:             (float) 0.5 min value to consider a slot requested

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`ontology.Ontology` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger` |.|

************************

'''

__author__ = "cued_dialogue_systems_group"

from policy import SummaryUtils
from scipy.stats import entropy
from ontology import Ontology
from utils import ContextLogger, Settings
logger = ContextLogger.getLogger('')

REQUESTING_THRESHOLD = 0.5


'''
#####Belief state related methods.####
'''

def arraySlotSummaryRel(belief, domainString):
    '''
    Gets the summary vector for goal slots, including the top probabilities, entropy, etc.

    :param belief: dict representing the full belief state
    :param domainString: string representing the domain
    :return: (dict) of slot goal summaries
    '''
    summary = {}
    slots = Ontology.global_ontology.get_common_slots_for_all_types(domainString)
        
    for domain in slots:
        for s in slots[domain]:
            slot = domain+'_'+s
            summary[slot] = {}
            slot_belief = belief['beliefs'][slot]
            summary[slot]['TOPHYPS'], summary[slot]['ISTOPNONE'] = SummaryUtils.getTopBeliefsExcludingNone(belief['beliefs'][slot])
            belief_dist = slot_belief.values()
            summary[slot]['ENTROPY'] = entropy(belief_dist)
            summary[slot]['ISREQUESTTOP'] = belief['beliefs']['requested'][slot] > 0.5

    return summary

def arraySlotSummaryRelPlain(belief, eType1, eType2):
    '''
    Gets the summary vector for goal slots, including the top probabilities, entropy, etc.

    :param belief: dict representing the full belief state
    :param domainString: string representing the domain
    :return: (dict) of slot goal summaries
    '''
    summary = {}
    slots = Ontology.global_ontology.get_common_slots(eType1, eType2)
        
    for s in slots:
        slot = s
        summary[slot] = {}
        slot_belief = belief['beliefs'][slot]
        summary[slot]['TOPHYPS'], summary[slot]['ISTOPNONE'] = SummaryUtils.getTopBeliefsExcludingNone(belief['beliefs'][slot])
        belief_dist = slot_belief.values()
        summary[slot]['ENTROPY'] = entropy(belief_dist)
        summary[slot]['ISREQUESTTOP'] = belief['beliefs']['requested'][slot] > 0.5

    return summary

# needs to be overridden as inform act is slot-independent
def getRequestedSlots(belief):
    '''
    Iterate get the list of mentioned requested slots

    :param belief: dict representing the full belief state
    :return: (list) of slot names with prob retrieved from belief > REQUESTING_THRESHOLD (an internal global)
    '''
    requested_slots = [],[] # non-rel, rel
    for slot in belief['beliefs']['requested']:
        requestprob = belief['beliefs']['requested'][slot]
        if requestprob > REQUESTING_THRESHOLD:
            if '#' in slot:
                requested_slots[1].append(slot)
            else:
                requested_slots[0].append(slot)
    return requested_slots

def getInformRequestedSlots(requested_slots, name, domainString):
    '''
    Informs about the requested slots from the last informed venue of form the venue informed by name

    :param requested_slots: list of requested slots
    :param name: name of the last informed venue
    :param domainString: string representing the domain
    :return: string representing the inform dialogue act
    '''
    result = Ontology.global_ontology.entity_by_features(domainString, {'name': name})

    if len(result) > 0:
        ent = result[0]
        return _getInformRequestedSlotsForEntity(requested_slots, ent, domainString)
    else:
        if not name:
            # Return a random venue
            result = []
            while len(result) == 0:
                rand_name = Ontology.global_ontology.getRandomValueForSlot(domainString, 'name', nodontcare=True)
                result = Ontology.global_ontology.entity_by_features(domainString, {'name': rand_name})
            ent = result[0]
            return _getInformRequestedSlotsForEntity(requested_slots, ent, domainString)

        else:
            logger.warning('Couldn\'t find the provided name: ' + name)
            return SummaryUtils.getInformNoneVenue({'name': name})
        
def _getInformRequestedSlotsForEntity(requested_slots, ent, domainString):
    '''
    Converts the list of requested slots and the entity into a inform_requested dialogue act

    :param requested_slots: double list of requested slots (obtained in getRequestedSlots()), first list contains non-rel slots, second list containts rel slots
    :param ent: dictionary with information about a database entity
    :return: string representing the dialogue act
    '''

    requested_slots = list(requested_slots)
    slotvaluepair = ['name="{}"'.format(ent['name'])]
    if len(requested_slots[0]) == 0 and len(requested_slots[1]) == 0:
        if 'type' in ent:
            slotvaluepair.append('type="{}"'.format(ent['type']))
        else:
            # type is not part of some ontologies. in this case just add a random slot-value
            slots = Ontology.global_ontology.get_requestable_slots(domainString)
            if 'name' in slots:
                slots.remove('name')
            slot = slots[Settings.random.randint(len(slots))]
            slotvaluepair.append('{}="{}"'.format(slot, ent[slot]))

    else:
        max_num_feats = 5
        if Settings.config.has_option("summaryacts", "maxinformslots"):
            max_num_feats = int(Settings.config.get('summaryacts', 'maxinformslots'))

        if len(requested_slots[0]) + len(requested_slots[1]) > max_num_feats:
            if len(requested_slots[0]) > max_num_feats:
                Settings.random.shuffle(requested_slots[0])
                requested_slots[0] = requested_slots[0][:max_num_feats]
                requested_slots[1] = []
            else:
                max_num_feats = max_num_feats - len(requested_slots[0])
                if len(requested_slots[1]) > max_num_feats:
                    Settings.random.shuffle(requested_slots[1])
                    requested_slots[1] = requested_slots[1][:max_num_feats]

        for slot in requested_slots[0]:
            if slot != 'name' and slot != 'location':
                if slot in ent:
                    slotvaluepair.append('{}="{}"'.format(slot, ent[slot]))
                else:
                    slotvaluepair.append('{}=none'.format(slot))
                    
        for slot in requested_slots[1]:
            # for now do nothing. respont not to requests for relations as access to other entities currently not possible from this code
            pass #entity, slot = slot.split('#')[0].split('_')

    return 'inform({})'.format(','.join(slotvaluepair))

#END OF FILE
