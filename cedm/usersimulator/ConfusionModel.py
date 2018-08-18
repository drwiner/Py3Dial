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
Created on 15 Jan 2018

@author: su259
'''
import usersimulator.ConfusionModel as cm
from ontology import Ontology
from utils import Settings
from utils import DiaAct as da
from cedm.utils.DActEntity import DactItemEntity
import numpy as np


class CERLevenshteinConfusionModel(cm.EMLevenshteinConfusionModel):
    def __init__(self, domainString):
        super(CERLevenshteinConfusionModel, self).__init__(domainString)
        
        self.slot_domain_relations = {}
        self.domain_slot_relations = Ontology.global_ontology.get_common_slots_for_all_types(self.domainString)
        for domain in self.domain_slot_relations:
            for slot in self.domain_slot_relations[domain]:
                s = slot.split('#')[0]
                if s not in self.slot_domain_relations:
                    self.slot_domain_relations[s] = set()
                self.slot_domain_relations[s].add(domain)
        
        self.slot_value_confusions = {}
        for slot in Ontology.global_ontology.get_system_requestable_slots(self.domainString) + [unicode('name')]:
            word_list = Ontology.global_ontology.get_informable_slot_values(self.domainString, slot) + [unicode('dontcare')]
            if slot in self.slot_domain_relations:
                word_list += [unicode('same')]
            self.slot_value_confusions[slot] = self.get_confusion_distributions(word_list, offset=0.15)
            if slot in self.slot_domain_relations:
                print self.slot_value_confusions[slot]

    def _get_confused_value_for_slot(self, slot, old_val):
        '''
        Randomly select a slot value for the given slot s different from old_val.
        '''
        if slot not in self.slot_value_confusions.keys() or old_val not in self.slot_value_confusions[slot]:
            return self._getRandomValueForSlotWithSame(slot, [old_val])
        else:
            return Settings.random.choice(self.slot_value_confusions[slot][old_val]['wlist'], p=self.slot_value_confusions[slot][old_val]['dist'])

    def _confuse_type(self, hyp):
        '''
        Create a wrong hypothesis, where the dialogue act type is different.
        '''
        hyp.items = []
        hyp.act = self._confuse_dia_act_type(hyp.act)
        item_format = da.actTypeToItemFormat[hyp.act]
        if item_format == 0:
            return hyp
        elif item_format == 1:
            new_slot_name = Ontology.global_ontology.get_random_slot_name(self.domainString)
            hyp.append(new_slot_name, None)
        elif item_format == 2:
            new_slot_name = Ontology.global_ontology.get_random_slot_name(self.domainString)
            assert new_slot_name is not None
            new_slot_val = self._getRandomValueForSlotWithSame(new_slot_name)
            if new_slot_val is None:
                hyp.append(new_slot_name, None)
            else:
                item = DactItemEntity(new_slot_name,'=',new_slot_val,self.domainString)
                item = self._transformSameToRel(item)
                hyp.addItem(item)
        # TODO: If item_format is 3, it doesn't confuse slot-values.
        # This might be a bug in the original implementation.
        return hyp

    def _confuse_slot(self, hyp):
        '''
        Create a wrong hypothesis, where the slot names are different.
        '''
        for dip in hyp.items:
            # If the slot is empty, just break
            if dip.slot is None:
                break

            slot = dip.slot
            if slot == 'more' or slot == 'type':
                break

            dip.slot = self._confuse_slot_name(slot)
            if dip.val is not None:
                dip.val = self._getRandomValueForSlotWithSame(slot=dip.slot)
                if dip.val is not None:
                    dip = self._transformSameToRel(dip)
#                 if 'same' in dip.val:
#                     # create relation
#                     dip.val = dip.slot
#                     dip.val = Settings.random.choice(self.slot_domain_relations[dip.slot])

        return hyp

    def _confuse_value(self, a_u):
        '''
        Create a wrong hypothesis, where one slot value is different.
        '''
        rand = Settings.random.randint(len(a_u.items))
        a_u_i = a_u.items[rand]

        if a_u_i.slot is not None and a_u_i.val is not None and a_u_i.slot != 'type':           
            old_value = a_u_i.val
            if isinstance(a_u_i,DactItemEntity) and a_u_i.is_relation():
                old_value = unicode('same')
                # ideally allow other relations with other domains. but in practice for my experiment, this is not necessary
#                 if len(self.slot_domain_relations[a_u_i.slot]) > 0:
#                     # means there are relations with other domains possible
#                     old_value = None
#                 else:
#                     old_value = unicode('same')
            
            
            a_u.items[rand].val = self._get_confused_value_for_slot(a_u_i.slot, old_value)
            
            if 'same' in a_u.items[rand].val:
                a_u.items[rand] = self._transformSameToRel(a_u.items[rand])
            elif isinstance(a_u.items[rand],DactItemEntity) and a_u.items[rand].is_relation():
                # remove relation status
                a_u.items[rand].value_entity = None

        return a_u
    
    def _getRandomValueForSlotWithSame(self, slot, not_these = []):
        if slot not in Ontology.global_ontology.get_informable_slots(self.domainString):
            return None
        candidate = Ontology.global_ontology.get_informable_slot_values(self.domainString, slot)
        candidate += ['dontcare']
        if slot in self.slot_domain_relations:
            #print "slot {} found in slot_domain_relations, same added"
            candidate += ['same']
        candidate = list(set(candidate) - set(not_these))
        # TODO - think should end up doing something like if candidate is empty - return 'dontcare' 
        if len(candidate) == 0:
            return 'dontcare'
    
        return Settings.random.choice(candidate)
    
    def _transformSameToRel(self, item):
        if 'same' == item.val:
            if not isinstance(item, DactItemEntity):
                item = DactItemEntity(item.slot,item.op,item.val,self.domainString)
            # create relation
            item.val = item.slot
            if item.slot == 'name':
                print "Woah"
            
            
            item.value_entity = Settings.random.choice(list(self.slot_domain_relations[item.slot]))
            
            if item.slot_entity == item.value_entity:
                print "Self-relation in confusion: entities: {}, slot_domain_relations {}".format(item.slot_entity,self.slot_domain_relations)
        return item
    
    def get_confusion_distributions_slots(self, word_list, offset=0.15):
        '''

        :param word_list: The list of words to be confused
        :param offset: Distribution softening factor, the largest the softer the distribution will be
        :return: dictionary
        '''
        wlist = list(word_list)
        Settings.random.shuffle(wlist)
        distributions = {}
        distances = [[self.levenshteinDistance(w1,w2) for w1 in wlist] for w2 in wlist]
        for i in range(len(wlist)):
            word = wlist[i]
            distributions[word] = {}
            sorted_indexes = np.argsort(distances[i])[1:self.len_confusion_list+1]
            sorted_wordlist = np.array(wlist)[sorted_indexes]
            distribution = np.array(distances[i])[sorted_indexes]
            distribution = 1./distribution
            distribution /= sum(distribution)
            distribution += offset
            distribution /= sum(distribution)
            distributions[word]['wlist'] = sorted_wordlist
            distributions[word]['dist'] = distribution
        return distributions
