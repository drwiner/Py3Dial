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
GoalGenerator.py - goal, agenda inventor for sim user 
====================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 
    
    import :mod:`utils.DiaAct` |.|
    import :mod:`utils.dact` |.|
    import :mod:`usersimulator.UMHdcSim` |.|
    import :mod:`ontology.Ontology` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger`

************************

'''

__author__ = "cued_dialogue_systems_group"

from usersimulator import UserModel as UM
from ontology import Ontology
from utils import ContextLogger, Settings
from cedm.utils import DActEntity as dact
import copy
logger = ContextLogger.getLogger('')

class GoalGenerator(UM.GoalGenerator):
    
    def __init__(self, dstring):
        super(GoalGenerator, self).__init__(dstring)
        
        self.conditional_constraints = {}
        self.conditional_constraints_slots = {}
    
    def init_goal(self, otherDomainsConstraints, um_patience, goal_with_rel):
        '''
        Initialises the goal g with random constraints and requests

        :param otherDomainsConstraints: of constraints from other domains in this dialog which have already had goals generated.
        :type otherDomainsConstraints: list
        :param um_patience: the patiance value for this goal
        :type um_patience: int
        :returns: (instance) of :class:`UMGoal`
        '''
        
        # clean/parse the domainConstraints list - contains other domains already generated goals:
        self._set_other_domains_constraints(otherDomainsConstraints)
        
        # Set initial goal status vars
        goal = CEGoal(um_patience, domainString=self.dstring)
        logger.debug(str(goal))
        num_attempts_to_resample = 2000
        
        commonSlots = Ontology.global_ontology.get_common_slots_for_all_types(self.dstring)
        commonSlots = {key: [elem.split('#')[0] for elem in value] for (key, value) in commonSlots.iteritems()}
        
        while True:
            num_attempts_to_resample -= 1
            # Randomly sample a goal (ie constraints):
            if not goal_with_rel or not otherDomainsConstraints:
                self._init_consts_requests(goal, um_patience)
                if goal.contains_rel(otherDomainsConstraints):
                    continue
                
                if goal_with_rel:
                    # if relations are requested make sure that first goal contains constraints which match slots of other domains
                    commonSlotFoundForDomain = dict.fromkeys(commonSlots.keys(), False)
                    for odom in commonSlots:
                        odomConstraints = []
                        for const in goal.constraints:
                            # make sure that slots and values match (eg do not allow dontcare
                            if const.slot in commonSlots[odom] and Ontology.global_ontology.is_value_in_slot(odom, value=const.val, slot=const.slot):
                                odomConstraints.append(copy.deepcopy(const))
                                commonSlotFoundForDomain[odom] = True
                                
                        if commonSlotFoundForDomain[odom]:
#                             # for debugging only
#                             isCandidate = 0
#                             for const in odomConstraints:
#                                 if const.slot == 'pricerange' and const.val == 'moderate':
#                                     isCandidate += 1
#                                 elif const.slot == 'area' and const.val == 'centre':
#                                     isCandidate += 1
#                             if isCandidate == 2:
#                                 print 'candidate found'
                            # also check if there is a solution in the other domain
                            odomMatches = Ontology.global_ontology.entity_by_features(odom, constraints=odomConstraints)
                            if not self.MIN_VENUES_PER_GOAL <= len(odomMatches) <= self.MAX_VENUES_PER_GOAL:
                                commonSlotFoundForDomain[odom] = False
                    if not any([True for x in commonSlotFoundForDomain.values() if x]):
                        continue
                
            else:
                self._init_consts_requests_rel(goal, um_patience)
                if not goal.contains_rel(otherDomainsConstraints):
                    continue
                
            # Check that there are venues that satisfy the constraints:
            venues = Ontology.global_ontology.entity_by_features(self.dstring, constraints=goal.constraints)
            #logger.info('num_venues: %d' % len(venues))
            if self.MIN_VENUES_PER_GOAL <= len(venues) <= self.MAX_VENUES_PER_GOAL:
                logger.dial('{}: GOAL FOUND'.format(self.dstring))
                break
#             if num_attempts_to_resample % 100 == 0:
#                 sys.stdout.write('.')
            if num_attempts_to_resample == 0:
                logger.error('Maximum number of goal resampling attempts reached.')
            
                
        if self.CONDITIONAL_BEHAVIOUR:
            # now check self.generator.conditional_constraints list against self.goal -assume any values that are the same
            # are because they are conditionally copied over from earlier domains goal. - set self.goal.copied_constraints
            goal.set_copied_constraints(all_conditional_constraints=self.conditional_constraints)

        # logger.warning('SetSuitableVenues is deprecated.')
        return goal
    
    def _init_consts_requests(self, goal, um_patience):
        '''
        Randomly initialises constraints and requests of the given goal.
        '''
        goal.clear(um_patience, domainString=self.dstring)

        # Randomly pick a task: bar, hotel, or restaurant (in case of TownInfo)
        goal.request_type = Ontology.global_ontology.getRandomValueForSlot(self.dstring, slot='type')

        # Get a list of slots that are valid for this task.
        valid_const_slots = Ontology.global_ontology.getValidSlotsForTask(self.dstring)
        
        # First randomly sample some slots from those that are valid: 
        sampling_probs = Ontology.global_ontology.get_sample_prob(self.dstring, 
                                                                  candidate=valid_const_slots,
                                                                  conditional_values=[])
        random_slots = list(Settings.random.choice(valid_const_slots,
                                size=min(self.MAX_CONSTRAINTS, len(valid_const_slots)),
                                replace=False,
                                p=sampling_probs))


        # Now randomly fill in some constraints for the sampled slots:
        for slot in random_slots:
            conditional_values = []
            if slot in self.conditional_constraints:
                conditional_values=self.conditional_constraints[slot]
            goal.add_const(self.dstring, slot, Ontology.global_ontology.getRandomValueForSlot(self.dstring, slot=slot, 
                                                            nodontcare=False,
                                                            conditional_values=conditional_values))
        
        # Add requests. Assume that the user always wants to know the name of the place
        goal.requests['name'] = None
        
        if self.MIN_REQUESTS == self.MAX_REQUESTS:
            n = self.MIN_REQUESTS -1  # since 'name' is already included
        else:
            n = Settings.random.randint(low=self.MIN_REQUESTS-1,high=self.MAX_REQUESTS)
        valid_req_slots = Ontology.global_ontology.getValidRequestSlotsForTask(self.dstring)
        if n > 0 and len(valid_req_slots) >= n:   # ie more requests than just 'name'
            choosen = Settings.random.choice(valid_req_slots, n,replace=False)
            for reqslot in choosen:
                goal.requests[reqslot] = None
    
    
    def _init_consts_requests_rel(self, goal, um_patience):
        '''
        Randomly initialises constraints and requests of the given goal.
        '''
        
        goal.clear(um_patience, domainString=self.dstring)

        # Randomly pick a task: bar, hotel, or restaurant (in case of TownInfo)
        goal.request_type = Ontology.global_ontology.getRandomValueForSlot(self.dstring, slot='type')

        # Get a list of slots that are valid for this task.
        valid_const_slots = Ontology.global_ontology.getValidSlotsForTask(self.dstring)
        
        # First randomly sample some slots from those that are valid: 
        sampling_probs = Ontology.global_ontology.get_sample_prob(self.dstring, 
                                                                  candidate=valid_const_slots,
                                                                  conditional_values=[])
        random_slots = list(Settings.random.choice(valid_const_slots,
                                size=min(self.MAX_CONSTRAINTS, len(valid_const_slots)),
                                replace=False,
                                p=sampling_probs))
        
        otherDomains = [key for key in self.conditional_constraints_slots]
        Settings.random.shuffle(otherDomains)
        
        for odom in otherDomains:
            # find shared slots and construct constraints
            constraints = {}
            for slot in Ontology.global_ontology.get_common_slots(self.dstring, odom):
                # N.B. could have used self.conditional_constraints_slots instead
                s1 = slot.split('#')[0]
                s2 = slot.split('#')[1] # often s1 and s2 are the same, e.g., area#area
                if s2 in self.conditional_constraints[odom] and \
                        any(self.conditional_constraints[odom][s2]) and \
                        Ontology.global_ontology.is_value_in_slot(self.dstring, self.conditional_constraints[odom][s2][0], s1):
                    constraints[s1] = self.conditional_constraints[odom][s2][0]
            
            
#             if not constraints:
#                 continue
            
            
            # Find all entities which match 
            matches = Ontology.global_ontology.entity_by_features(self.dstring, constraints)
            
            if any(matches):
                Settings.random.shuffle(matches)
                
                match = matches[0]
            
                # Check if at least one of the slots of the constraints from the other domain are in the selected subset of slots
                slotFound = False
                for cond_slot in constraints: 
                    if cond_slot in random_slots:
                        slotFound = True
                        break
                    
                # if not, add a random slot
                if not slotFound:
                    randSlot = Settings.random.choice(self.conditional_constraints_slots[odom])
                    random_slots.append(randSlot)
                    
                # Now fill in the sampled slots with the constraints of the found entity
                for slot in random_slots:
                    goal.add_const(self.dstring, slot, match[slot])
                    if slot in self.conditional_constraints_slots[odom]:
                        # now add relation constraint
                        goal.add_const_rel(self.dstring, slot, odom, slot)
                    
            
                # Add requests. Assume that the user always wants to know the name of the place
                goal.requests['name'] = None
                
                if self.MIN_REQUESTS == self.MAX_REQUESTS:
                    n = self.MIN_REQUESTS -1  # since 'name' is already included
                else:
                    n = Settings.random.randint(low=self.MIN_REQUESTS-1,high=self.MAX_REQUESTS)
                valid_req_slots = Ontology.global_ontology.getValidRequestSlotsForTask(self.dstring)
                if n > 0 and len(valid_req_slots) >= n:   # ie more requests than just 'name'
                    choosen = Settings.random.choice(valid_req_slots, n,replace=False)
                    for reqslot in choosen:
                        goal.requests[reqslot] = None
                        
                return
                
#     def _init_consts_requests_rel(self, goal, um_patience):
#         '''
#         Randomly initialises constraints and requests of the given goal.
#         '''
#         
#         goal.clear(um_patience, domainString=self.dstring)
# 
#         # Randomly pick a task: bar, hotel, or restaurant (in case of TownInfo)
#         goal.request_type = Ontology.global_ontology.getRandomValueForSlot(self.dstring, slot='type')
# 
#         # Get a list of slots that are valid for this task.
#         valid_const_slots = Ontology.global_ontology.getValidSlotsForTask(self.dstring)
#         
#         # First randomly sample some slots from those that are valid: 
#         sampling_probs = Ontology.global_ontology.get_sample_prob(self.dstring, 
#                                                                   candidate=valid_const_slots,
#                                                                   conditional_values=[])
#         random_slots = list(Settings.random.choice(valid_const_slots,
#                                 size=min(self.MAX_CONSTRAINTS, len(valid_const_slots)),
#                                 replace=False,
#                                 p=sampling_probs))
# 
# 
#         # Now randomly fill in some constraints for the sampled slots:
#         for slot in random_slots:
#             goal.add_const(self.dstring, slot, Ontology.global_ontology.getRandomValueForSlot(self.dstring, slot=slot, 
#                                                             nodontcare=False,
#                                                             conditional_values=[]))
#             
#         
#         # dealing with relations
#         relatedslots = set()
#         for domainstring in self.conditional_constraints_slots:
#             for cond_slot in self.conditional_constraints_slots[domainstring]: 
#                 
#                 # check if there is already a relation of that slot with a different domain
#                 if cond_slot not in relatedslots:
#                     # make sure goal has correct value for slot constraint
#                     if goal.contains_slot_const(cond_slot):
#                         goal.remove_slot_const(cond_slot)
#                     
#                     # assuming there is only one value in the constraints for slot cond_slot
#                     assert len(self.conditional_constraints[domainstring][cond_slot]) == 1
#                     
#                     val = self.conditional_constraints[domainstring][cond_slot][0]
#                     goal.add_const(self.dstring, cond_slot, val)
#                     
#                     # now add relation constraint
#                     goal.add_const_rel(self.dstring, cond_slot, domainstring, cond_slot)
#             
#                 
#             
#         
#         # Add requests. Assume that the user always wants to know the name of the place
#         goal.requests['name'] = None
#         
#         if self.MIN_REQUESTS == self.MAX_REQUESTS:
#             n = self.MIN_REQUESTS -1  # since 'name' is already included
#         else:
#             n = Settings.random.randint(low=self.MIN_REQUESTS-1,high=self.MAX_REQUESTS)
#         valid_req_slots = Ontology.global_ontology.getValidRequestSlotsForTask(self.dstring)
#         if n > 0 and len(valid_req_slots) >= n:   # ie more requests than just 'name'
#             choosen = Settings.random.choice(valid_req_slots, n,replace=False)
#             for reqslot in choosen:
#                 goal.requests[reqslot] = None
                
    def _set_other_domains_constraints(self, otherDomainsConstraints):
        """Simplest approach for now: just look for slots with same name 
        """
        # Get a list of slots that are valid for this task.
        for domainstring in otherDomainsConstraints:
            
            valid_const_slots = Ontology.global_ontology.getValidSlotsForTask(self.dstring) 
            self.conditional_constraints[domainstring] = {slot: [] for slot in valid_const_slots}
    
            if not self.CONDITIONAL_BEHAVIOUR:
                self.conditional_constraints_slots[domainstring] = []
                return
    
            for const in otherDomainsConstraints[domainstring]:
                if const.slot in valid_const_slots and const.val != "dontcare": #TODO think dontcare should be dealt with diff
                    # issue is that first domain may be dontcare - but 2nd should be generated conditioned on first.
                    if const.op == "!=":
                        continue
                    
                    #TODO delete: if const.val in Ontology.global_ontology.ontology['informable'][const.slot]: #make sure value is valid for slot
                    if Ontology.global_ontology.is_value_in_slot(self.dstring, value=const.val, slot=const.slot):
                        self.conditional_constraints[domainstring][const.slot] += [const.val]  
            self.conditional_constraints_slots[domainstring] = [s for s,v in self.conditional_constraints[domainstring].iteritems() if len(v)]
        return
                
class CEGoal(UM.UMGoal):
    def __init__(self, patience, domainString):
        super(CEGoal, self).__init__(patience, domainString)
        self.dstring = domainString
    
    def contains_rel(self, otherDomainConstraints = {}):
        for constraint in self.constraints:
            if constraint.is_relation():
                return True
        
        for oDomain in otherDomainConstraints:
            for oCons in otherDomainConstraints[oDomain]:
                selfVal = self.get_correct_const_value(oCons.slot)
                if selfVal == oCons.val:
                    return True
        
        return False
    
    def get_relation_split(self):
        relationList = []
        nonrelationList = []
        for constraint in self.constraints:
            if constraint.is_relation():
                relationList.append(constraint)
            else:
                nonrelationList.append(constraint)
                
        return relationList, nonrelationList
    
    def contains_slot_const(self, slot, eType = None):
        if eType is None:
            eType = self.dstring
        for item in self.constraints:
            # an error introduced here by dact.py __eq__ method: 
            #if item.slot == slot:
            if isinstance(item, dact.DactItemEntity):
                if str(item.slot) == slot and str(item.slot_entity == eType):
                        return True
                if item.is_relation():
                    if str(item.val) == slot and str(item.value_entity == eType):
                        return True
            else:
                if str(item.slot) == slot:
                    return True 
        return False
    
    def get_correct_const_value(self, slot, negate=False, eType = None):
        '''
        :return: (list of) value of the given slot in user goal constraint.

        '''
        values = []
        for item in self.constraints:
            if item.slot == slot and not item.is_relation() and (eType is None or eType == item.slot_entity):
                if item.op == '!=' and negate or item.op == '=' and not negate:
                    values.append(item.val)
        if len(values) == 1:
            return values[0]
        elif len(values) == 0:
            return None
        logger.error('Multiple values are found for %s in constraint: %s' % (slot, str(values)))
        return values

    def get_correct_const_rel(self, slot, eType = None):
        '''
        :return: (list of) value of the given slot in user goal constraint.

        '''
        relations = []
        for item in self.constraints:
            if item.slot == slot and item.is_relation() and (eType is None or eType == item.slot_entity):
                relations.append(copy.deepcopy(item))
            elif item.val == slot and item.is_relation() and (eType is None or eType == item.value_entity):
                relations.append(copy.deepcopy(item))

        if len(relations) == 1:
            return relations[0]
        elif len(relations) == 0:
            return None
        logger.error('Multiple relations are found for %s in constraint: %s' % (slot, str(relations)))
        return relations
    
            
    def add_const(self, eType, slot, value, negate=False):
        """
        """
        if not negate:
            op = '='
        else:
            op = '!='
        item = dact.DactItemEntity(slot, op, value, eType)
        self.constraints.append(item)
        
    def add_const_rel(self, eType1, slot1, eType2, slot2, negate=False):
        """
        """
        if not negate:
            op = '='
        else:
            op = '!='
        item = dact.DactItemEntity(slot1, op, slot2, eType1, eType2)
        self.constraints.append(item)
        
    def replace_const(self, slot, value, negate=False, eType=None):
        self.remove_slot_const(slot, negate)
        self.add_const(eType, slot, value, negate)
        
    def remove_slot_const(self, slot, negate=None, eType = None, relOnly=False):
        if eType is None:
            return super(CEGoal, self).remove_slot_const(slot, negate)
        
        copy_consts = copy.deepcopy(self.constraints)
        
        if negate is not None:
            if not negate:
                op = '='
            else:
                op = '!='
        
            for item in copy_consts:
                if item.slot == slot:
                    if item.op == op:
                        self.constraints.remove(item)
        else:
            for item in copy_consts:
                if item.is_relation():
                    if item.slot == slot and item.slot_entity == eType:
                        self.constraints.remove(item)
                    elif item.val == slot and item.value_entity == eType:
                        self.constraints.remove(item)
                elif item.slot == slot and not relOnly:
                    self.constraints.remove(item)
                

    def is_satisfy_all_consts(self, item):
        '''
        Check if all the given items set[(slot, op, value),..] satisfies all goal constraints (conjunction of constraints).
        '''
        if type(item) is not set:
            item = set([item])
        for it in item:
            for const in self.constraints:
                if not const.match(it):
                    return False
        return True

    def add_name_constraint(self, value, negate=False):
        if value in [None, 'none']:
            return

        wrong_venues = self.get_correct_const_value_list('name', negate=True)
        correct_venue = self.get_correct_const_value('name', negate=False)

        if not negate:
            # Adding name=value but there is name!=value.
            if value in wrong_venues:
                logger.error('Failed to add name=%s: already got constraint name!=%s.' %
                             (value, value))
                return
            
            # Can have only one name= constraint.
            if correct_venue is not None:
                #logger.debug('Failed to add name=%s: already got constraint name=%s.' %
                #             (value, correct_venue))
                self.replace_const('name', value, eType = self.dstring) # ic340: added to override previously informed venues, to avoid
                                                    # simuser getting obsessed with a wrong venue
                return

            # Adding name=value, then remove all name!=other.
            self.replace_const('name', value, eType = self.dstring)
            return

        # if not negate and not self.is_suitable_venue(value):
        #     logger.debug('Failed to add name=%s: %s is not a suitable venue for goals.' % (value, value))
        #     return

        if negate:
            # Adding name!=value but there is name=value.
            if correct_venue == value:
                logger.error('Failed to add name!=%s: already got constraint name=%s.' % (value, value))
                return

            # Adding name!=value, but there is name=other. No need to add.
            if correct_venue is not None:
                return

            self.add_const(self.dstring, 'name', value, negate=True)
            return
    
    
    
