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
from cedm.utils.DActEntity import DactItemEntity
from copy import deepcopy
'''
UMHdcSim.py - Handcrafted simulated user behaviour 
====================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 
    
    import :mod:`usersimulator.UserModel` |.|
    import :mod:`utils.DiaAct` |.|
    import :mod:`utils.dact` |.|
    import :mod:`ontology.Ontology` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger`

**Relevant config variables** (values are defaults)::

    [um]
    usenewgoalscenarios = True 
    answerreqalways = False

    [goalgenerator]
    patience = 10

'''

__author__ = "cued_dialogue_systems_group"

from cedm.utils import DActEntity
from utils import ContextLogger, Settings, DiaAct, dact
from usersimulator import UMHdcSim, UserModel
from ontology import Ontology
import copy
logger = ContextLogger.getLogger('')


class UMHdcSimER(UMHdcSim.UMHdcSim):
    '''Handcrafted behaviour of simulated user
    '''
    def __init__(self, domainString, max_patience = 5, informState = {}):
        super(UMHdcSimER,self).__init__(domainString,max_patience)
        
        self.relProb = {}

        self.agenda = UMAgendaER(self.dstring)
        
        self.receive_options_rel = {'request': self._receive_request_rel,
                   'confirm': self._receive_confirm_rel,
                   'select': self._receive_select_rel
                   }
        
        self.COUNTER = 0
        
        self.informState = informState
        

    def init(self, goal, um_patience):
        """
        """
        # First create the AGENDA:
        self.goal = goal
        self.last_user_act = DActEntity.DiaActEntity('null()', self.dstring)
        self.last_sys_act = DActEntity.DiaActEntity('null()', self.dstring)
        
        #if self.sampleDecisiconProbs:
        if self.sampling_probs:
            self._sampleProbs()
            
        self.max_patience = um_patience
        
        self.relax_constraints = False
        self.first_venue_recommendation = True

        for const in goal.constraints:
            if not const.is_relation():
                slot = const.slot
                value = const.val
                goal.add_prev_used(slot, value)
        
        if self.dstring not in self.informState:
            self.informState[self.dstring] = DomainUserState()
        self.informState[self.dstring].reset(self.goal)
        self.agenda.initialized = False
#         print "Init new domain: {}".format(self.dstring)

    def receive(self, sys_act, goal):
        """
        """
        if not self.agenda.initialized:
            self._init_agenda()
            
#         print "informState is {} for domain {}: {}".format(id(self.informState),self.dstring,self.informState)
            
        
        if not isinstance(sys_act, DActEntity.DiaActEntity) or not sys_act.contains_relation(): 
            super(UMHdcSimER, self).receive(sys_act, goal)
        else:
            sys_act = self._normalizeSysAct(sys_act)
            
            self.last_sys_act = sys_act
    
            if goal.is_completed() and self.agenda.size() == 0 and sys_act.act != 'reqmore'\
                    and Settings.random.rand() < 0.85:
                # Goal already completed: say goodbye.
                self.agenda.clear()
                self.agenda.push(DActEntity.DiaActEntity('bye()', self.dstring))
                return
    
            # Generate repeat act with small probability:
            #   assume the user did not hear the system utterance,
            #   let alone make any updates to their (user goal) state,
            #   and respond with a repeat act.
            if goal.patience > 1 and sys_act.act != 'repeat' and sys_act.act != 'badact' and\
                            sys_act.act != 'null':
                if Settings.random.rand() < self.rand_decision_probs['Repeat']:
                    self.agenda.push(DActEntity.DiaActEntity('repeat()', self.dstring))
                    return
    
            # Generate null action with small probability:
            #   user generates (silence or) something incomprehensible
            if Settings.random.rand() < self.rand_decision_probs['NullResp']:
                self.agenda.push(DActEntity.DiaActEntity('null()', self.dstring))
                return
    
            if sys_act.act in self.receive_options and sys_act.act != 'null': 
                self.receive_options_rel[sys_act.act](sys_act, goal)
            else:
                logger.warning('Unknown acttype for relations in UMHdcSimER.receive(): ' + sys_act.act)
                self._receive_badact(goal)
    
            logger.debug(str(self.agenda.agenda_items))
            logger.debug(str(goal))

    def respond(self, goal):
        '''
        This method is called to get the user response.

        :param goal: of :class:`UMGoal` 
        :type goal: instance
        :returns: (instance) of :class:`DiaActWithProb`
        '''
        
        if not self.agenda.initialized:
            self._init_agenda()
        
        # If agenda is empty, push ByeAct on top.
        if self.agenda.size() == 0:
            self.agenda.push(DActEntity.DiaActEntity('bye()', self.dstring))

        # Pop the top act off the agenda to form the user response.
        dap = self.agenda.pop()
        logger.debug(str(dap))

        # if len(dap.items) > 1:
        #     logger.warning('Multiple semantic items in agenda: ' + str(dap))
        dap_item = None
        if len(dap.items) > 0:
            dap_item = dap.items[0]

        # If it created negate(name="!x") or deny(name="x", name="!x") or confirm(name="!x") just reqalts()
        for item in dap.items:
            if item.op == "!=":
                dap = DActEntity.DiaActEntity('reqalts()', self.dstring)
                break

        # Checking agenda for redundant constraints.
        self.agenda.filter_constraints(dap)

        if dap.act in ['thankyou', 'silence', 'repeat', 'ack', 'deny', 'confirm']:
            return self._enforce_DiaActEntity(self._normalise_act_no_rules(dap))

        if self.last_sys_act.act == 'reqmore':
            return self._enforce_DiaActEntity(self._normalise_act_no_rules(dap))

        # Ckecing whether we might remove the slot name for value dontcare in the planned act.
        if dap.act == 'inform' and not dap.items:
            logger.error('Error inform act with no slots is on agenda.')

        # In response to a request about a particular slot users often do not specify hte slot
        # especially when the value is dontcare.
        if self.last_sys_act.act in ['request', 'confreq', 'select']:
            if dap.act == 'inform' and dap_item is not None and dap_item.val == 'dontcare':
                f = Settings.random.rand()
                if f < self.rand_decision_probs['NoSlotWithDontcare']:
                    dap_item.slot = None

        # Checking whether we might add a venue name ot the planned act.
        if dap.act == 'request' and len(dap.items) == 1:
            rec_ven = goal.requests['name']
            # If venue recommended, randomly decide to include the venue name in the request.
            if rec_ven is not None:
                if Settings.random.rand() < self.rand_decision_probs['AddVenueNameToRequest']:
                    dap.append('name', rec_ven)
            # else:
            #     logger.error('Requesting slot without venue recommended.')

        # Checking whether we might include additional constraints in the planned act.
        # When specifying a constraint, combine the act with additional constraints with some probability.
        if dap.act in ['inform', 'negate', 'hello', 'affirm']:
#             print "dialogue act", dap.act, dap.items
            inf_comb_count = 0
            while self.agenda.size() > 0 and \
                    (self.agenda.agenda_items[-1].act == 'inform' or \
                     self.agenda.agenda_items[-1].act == 'request' and dap.act == 'hello'):
                if Settings.random.rand() < self.rand_decision_probs['InformCombination']:
                    inf_comb_count += 1
                    next_dap = self.agenda.pop()
                    for dip in next_dap.items:
                        dap.append(dip.slot, dip.val, dip.op == '!=')
                else:
                    break

        # Checking whether we might request a slot when specifying the type of venue.
        # When specifying the requestType constraint at the beginning of a dialogue,
        # occasionally request an additional requested slot
        if dap.act == 'request' and len(dap.items) > 0 and dap_item.slot in ['type', 'task', 'restaurant']:
            logger.warning('Not completely implemented: RequestSlotAtStart')

        usr_output = self._normalise_act_no_rules(dap)
        self.last_user_act = usr_output
        return self._enforce_DiaActEntity(usr_output)
    
    def _normalizeSysAct(self, sys_act):
        if sys_act.contains_relation():
            new_act = deepcopy(sys_act)
            for item in new_act.items:
                if item.is_relation() and item.slot_entity != self.dstring:
                    if item.value_entity != self.dstring:
                        logger.error('item does not refer to this entity / domain: {}'.format(item))
                    
                    # switch entities and slots
                    store_entity = item.value_entity
                    item.value_entity = item.slot_entity
                    item.slot_entity = store_entity
                    
                    store_slot = item.val
                    item.val = item.slot
                    item.slot = store_slot
            logger.dial("Item {} normalized to {}".format(sys_act,new_act))
            return new_act
        else:
            return sys_act
    
    def _enforce_DiaActEntity(self, dact):
        if not isinstance(dact, DActEntity.DiaActEntity):
            return DActEntity.DiaActEntity(dact,self.dstring)
        return dact
    
    def _receive_request(self, sys_act, goal):
        items = sys_act.items
        requested_slot = items[0].slot

        # Check if any options are given.
        if len(items) > 1:
            logger.error('request(a,b,...) is not supported: ' + sys_act)

        '''
        First check if the system has actually already recommended the name of venue.
        If so, check if the user is still trying to get requested info.
        In that case, don't respond to the request (in at least some of the cases)
        but ask for the requested info.
        '''
         
        # Check if there is an unsatisfied request on the goal
        if 'name' in goal.requests and goal.requests['name'] is not None:
            for info in goal.requests:
                if goal.requests[info] is None:
                    self.agenda.push(DActEntity.DiaActEntity('request(name="%s",%s)' % (goal.requests['name'], info), self.dstring))
                    return

        '''
        request(venue), request(task), ...
        Check if there is an act of the form request(bar|restaurant|hotel) on the agenda.
        If so, then jus return, otherwise push that act onto the agenda
        '''
        
        if requested_slot in ['venue', 'task', 'type']:
            # Check if there is a suitable response on the agenda (eg. request(bar)).
            if self.agenda.contains_act('request'):
                # :todo: this might be problem. because now any request act on the agenda trigger a return!
                return
            self.agenda.push(DActEntity.DiaActEntity('inform(type="%s")' % goal.request_type, self.dstring))
            return

        '''
        request(info)
        "Do you know the phone number of the place you are looking for?", etc.
        Just say no.
        '''
       
        if Ontology.global_ontology.is_only_user_requestable(self.dstring, slot=requested_slot):
            self.agenda.push(DActEntity.DiaActEntity('negate()', self.dstring))
            return

        '''
        request(hotel), request(bar), request(restaurant)
        This type of system action is not produced by handcrafted DM.
        '''

        '''
        Handle invalid requests that do not match the user goal
        '''
        # Check if the requested slot makes no sense for the user's query,
        # eg. system asks for "food" when user requests "hotel".
        # In this case, re-request the type.
        if not Ontology.global_ontology.is_valid_request(self.dstring, request_type=goal.request_type, slot=requested_slot): 
            self.agenda.push(DActEntity.DiaActEntity('inform(type="%s")' % goal.request_type, self.dstring))
            return

        '''
        Handle valid requests
        '''
        requested_value = goal.get_correct_const_value(requested_slot)
        requested_relation = goal.get_correct_const_rel(requested_slot)
        
        answer_slots = [requested_slot]
        
        '''
        Case 1: Requested slot is somewhere on the agenda.
        '''
        # Go through the agenda and locate any corresponding inform() acts.
        # If you find one, move it to the top of the agenda.
        
        logger.debug("CASE1")
        action_taken = False
        inform_items = self.agenda.get_agenda_with_act('inform')
        for agenda_act in inform_items:
            for item in agenda_act.items:
                if item.slot in answer_slots:
                    # Found corresponding inform() on agenda and moving it to top.
                    action_taken = True
                    self.agenda.remove(agenda_act)
                    self.agenda.push(agenda_act)

        if action_taken:
            logger.debug('CASE1')
            return

        '''
        Case 2: Requested slot is not on the agenda, but there is another request() or inform() on the agenda.
        '''
        
        logger.debug("CASE2")
        if not self.answer_req_always:
            if self.agenda.get_agenda_with_act('inform') != [] or self.agenda.get_agenda_with_act('request') != []:
                logger.debug('CASE2')
                return

        '''
        Case 3: There is nothing on the agenda that would suit this request,
                but there is a corresponding constraint in the user goal.
        '''
        logger.debug("CASE3")
        if goal.contains_slot_const(requested_slot, self.dstring):
            logger.debug('CASE3')
            new_act = DActEntity.DiaActEntity('inform()', self.dstring)
            self.COUNTER += 1
            
            if requested_relation is not None and requested_value is not None and Settings.random.rand() < self.relProb.get(self.agenda._getOtherObject(requested_relation), 0.0):
                new_act.addItem(requested_relation)
            elif requested_value is not None:
                new_act.append(requested_slot, requested_value)
                
            wrong_val = goal.get_correct_const_value(requested_slot, negate=True)
            if wrong_val is not None:
                new_act.append(requested_slot, wrong_val, negate=True)
            self.agenda.push(new_act)
            return
         
        '''
        Case 4: There is nothing on the agenda or on the user goal.
        '''
        logger.debug('###4 ---- into case 4 --- prob going to say dontcare ...')
        # Either repeat last user request or invent a value for the requested slot.
        f = Settings.random.rand()
        if f < self.rand_decision_probs['NewRequestResp1']:
            # Decided to randomly repeat one of the goal constraints.
            # Go through goal and randomly pick a request to repeat.
            random_val = 'dontcare' #Ontology.global_ontology.getRandomValueForSlot(self.dstring, requested_slot, True) 
            goal.get_correct_const_value(requested_slot) # copied here from below because random_val was not defined. IS THIS CORRECT?
            if len(goal.constraints) == 0:
                # No constraints on goal: say dontcare.
                self.agenda.push(DActEntity.DiaActEntity('inform(=dontcare)', self.dstring))
                goal.add_const(eType=self.dstring,slot=requested_slot,value=random_val)
                goal.add_prev_used(requested_slot,random_val)
                logger.debug('###4.1 just added to goal.prev_slot_values '+ str(requested_slot)+' '+str(random_val))
            else: 
                logger.debug('###4.2')
                sampled_act = Settings.random.choice(goal.constraints)
                new_act = DActEntity.DiaActEntity('inform()', self.dstring)
                new_act.addItem(sampled_act)
                self.agenda.push(new_act)

        elif f < self.rand_decision_probs['NewRequestResp1'] + self.rand_decision_probs['NewRequestResp2']:
            # Pick a constraint from the list of options and randomly invent a new constraint.
            #random_val = goal.getCorrectValueForAdditionalConstraint(requested_slot) # wrong method from dongho?
            random_val = Ontology.global_ontology.getRandomValueForSlot(self.dstring, requested_slot, True) 
            if random_val is None:
                # TODO  Not sure what options is meant to be/do? --> will just reply 'dontcare' for 
                self.agenda.push(DActEntity.DiaActEntity('inform(=dontcare)', self.dstring))
                goal.add_const(eType=self.dstring,slot=requested_slot,value=random_val)
                goal.add_prev_used(requested_slot,random_val)
                logger.debug('###4.1 just added to goal.prev_slot_values '+ str(requested_slot)+' '+str(random_val))
            else:
                goal.add_const(eType=self.dstring, slot=requested_slot,value=random_val) 
                #goal.constraints[requested_slot] = random_val
                self.agenda.push(DActEntity.DiaActEntity('inform(%s="%s")' % (requested_slot, random_val), self.dstring))
                logger.debug('###4.5 -- havent added anything to prev_slot_values')
    
        else:
            # Decided to say dontcare. 
            logger.debug('###4.6')
            self.agenda.push(DActEntity.DiaActEntity('inform(=dontcare)', self.dstring))
            goal.add_const(eType=self.dstring,slot=requested_slot,value='dontcare')
            goal.add_prev_used(requested_slot,'dontcare')
            logger.debug('###4.1 just added to goal.prev_slot_values '+ str(requested_slot)+' '+str('dontcare'))

    def _receive_confirm_rel(self, sys_act, goal):
        self._checkIfRelStillHolds()
        
        # Check the given information.
        if not self._receive_implicit_confirm_rel(sys_act, goal):
            # The given info was not ok, so stop processing confirm act.
            return

        # Check explicit confirmation.
        if not self._receive_direct_implicit_confirm_rel(sys_act, goal):
            # Item in system act needed correction: stop processing confirm act here.
            return

        # Given information is ok. Put an affirm on the agenda if next item on the agenda is an inform act,
        # can affirm and inform in one go.
        # affirm(), or affirm(a=x) if next agneda item is inform.
        new_affirm_act = DActEntity.DiaActEntity('affirm()', self.dstring)
        if self.agenda.size() > 0:
            top_item = self.agenda.agenda_items[-1]
            if top_item.act == 'inform':
                if Settings.random.rand() < self.rand_decision_probs['AffirmCombination']:
                    for item in top_item.items:
                        new_affirm_act.addItem(item)
                    self.agenda.pop()
        self.agenda.push(new_affirm_act)
        
    def _receive_confirm(self, sys_act, goal):
        # Check the given information.
        if not self._receive_implicit_confirm(sys_act, goal):
            # The given info was not ok, so stop processing confirm act.
            return

        # Check explicit confirmation.
        if not self._receive_direct_implicit_confirm(sys_act, goal):
            # Item in system act needed correction: stop processing confirm act here.
            return

        # Given information is ok. Put an affirm on the agenda if next item on the agenda is an inform act,
        # can affirm and inform in one go.
        # affirm(), or affirm(a=x) if next agneda item is inform.
        new_affirm_act = DActEntity.DiaActEntity('affirm()', self.dstring)
        if self.agenda.size() > 0:
            top_item = self.agenda.agenda_items[-1]
            if top_item.act == 'inform':
                if Settings.random.rand() < self.rand_decision_probs['AffirmCombination']:
                    for item in top_item.items:
                        new_affirm_act.addItem(item)
                    self.agenda.pop()
        self.agenda.push(new_affirm_act)
        
    def _receive_implicit_confirm_rel(self, sys_act, goal):
        '''
        This method is used for checking implicitly confirmed items. Currently only called from confirm => only one item in sys_act
        :param sys_act:
        :param goal:
        :return: True if all the items are consistent with the user goal.
                 If there is a mismatch, then appropriate items are added to the agenda and
                 the method returns False.
        '''
        if not sys_act.contains_relation():
            logger.error('Sys_act should contain a relation.')
            
        # Check if all the implicitly given information is correct. Otherwise reply with negate or deny.
        item = sys_act.items[0]
            
        if item.slot_entity == self.dstring:
            eType = item.slot_entity
            slot = item.slot
            eType2 = item.value_entity
            slot2 = item.val
        elif item.value_entity == self.dstring:
            eType2 = item.slot_entity
            slot2 = item.slot
            eType = item.value_entity
            slot = item.val
            
        normItem = DactItemEntity(slot,'=',slot2,eType,eType2)
        
        correct_rel = None
        correct_val = None
        do_correct_misunderstanding = False
        # negation = (item.op == '!=')

        if slot in ['count', 'option']:
            return True

        # Exclude slots that are info keys in the ontology, go straight to the next item.
        if Ontology.global_ontology.is_only_user_requestable(self.dstring, slot=slot) and\
                not goal.contains_slot_const(slot):
            return True

        # If an implicitly confirmed slot is not present in the goal,
        # it doesn't really matter, unless:
        # a) the system claims that there is no matching venue, or
        # b) the system explicitly confirmed this slot.
        if not goal.contains_slot_const(slot):
            logger.debug('{} is not on user goal.',format(sys_act.items))
        else:
            correct_rel = goal.get_correct_const_rel(slot, eType = eType)
            correct_val = goal.get_correct_const_value(slot, eType = eType)
            
            # Conflict between user goal, and system act
            if correct_rel is None or (correct_rel is not None and not goal.is_satisfy_all_consts(normItem)):
                do_correct_misunderstanding = True
                    
            
        if do_correct_misunderstanding:
            logger.debug('Correct misunderstanding for slot %s' % slot)
            # Depending on the patience level, say bye with some probability (quadratic function of patience level)
            if not self.patience_old_style:
                prob1 = float(goal.patience ** 2) / self.max_patience ** 2
                prob2 = float(2*goal.patience) / self.max_patience
                prob = prob1 - prob2 + 1
                if Settings.random.rand() < prob:
                    # Randomly decided to give up
                    self.agenda.clear()
                    self.agenda.push(DActEntity.DiaActEntity('bye()', self.dstring))
                    return False

            # Pushing negate or deny onto agenda to correct misunderstanding.
            # Make a random decision as to whether say negate(a=y) or deny(a=y,a=z), or
            # confirm a constraint not mentioned in system act.
            # If the type is wrong, say request(whatever).

            if slot != 'type':
                planned_response_act = None
                
#                 rel = "{}#{}".format(item.slot_entity,item.value_entity)
                
                
                f = Settings.random.rand()
                if f < self.rand_decision_probs['CorrectingAct1']:
                    planned_response_act = DActEntity.DiaActEntity('negate()', self.dstring)
                    if correct_rel is not None:
                        planned_response_act.addItem(correct_rel)
                    elif correct_val is not None:
                        planned_response_act.addItem(DactItemEntity(normItem.slot,'=',correct_val,normItem.slot_entity))
                elif f < self.rand_decision_probs['CorrectingAct1'] + self.rand_decision_probs['CorrectingAct2']:
                    planned_response_act = DActEntity.DiaActEntity('negate()', self.dstring)
                    if correct_rel is not None:
                        planned_response_act.addItem(correct_rel)
                    elif correct_val is not None:
                        planned_response_act.addItem(DactItemEntity(normItem.slot,'=',correct_val,normItem.slot_entity))
#                     planned_response_act.addItem(item)
                else:
                    planned_response_act = DActEntity.DiaActEntity('inform()', self.dstring)
                    if correct_rel is not None:
                        planned_response_act.addItem(correct_rel)
                    elif correct_val is not None:
                        planned_response_act.addItem(DactItemEntity(normItem.slot,'=',correct_val,item.slot_entity))
            else:
                planned_response_act = DActEntity.DiaActEntity('inform(type=%s)' % goal.request_type, eType )

            self.agenda.push(planned_response_act)

            # Resetting goal request slots.
            goal.reset_requests()
            return False

        # Implicit confirmations okay.
        return True
    
    def _receive_direct_implicit_confirm_rel(self, sys_act, goal):
        '''
        Deals with implicitly confirmed items that are not on the user goal.
        These are okay in system inform(), but not in system confirm() or confreq().
        In this case, the user should mention that slot=dontcare.
        :param sys_act:
        :param goal:
        :return:
        '''
        for item in sys_act.items:
            slot = item.slot
            val = item.val
            if slot in ['count', 'option', 'type']:
                continue
            if not goal.contains_slot_const(slot) and val != 'dontcare':
                self.agenda.push(DActEntity.DiaActEntity('negate(%s="dontcare")' % slot, self.dstring))
                return False

        # Explicit confirmations okay.
        return True
        
    def _receive_select_rel(self, sys_act, goal):
        # Receive select(slot=x, slot=y)
        if not sys_act.contains_relation():
            logger.error('Sys_act should contain a relation.')
            
        if sys_act.item[0].slot_entity == self.dstring:
            eType = sys_act.item[0].slot_entity
            slot = sys_act.item[0].slot
            eType2 = sys_act.item[0].value_entity
            slot2 = sys_act.item[0].value
        elif sys_act.item[0].value_entity == self.dstring:
            eType2 = sys_act.item[0].slot_entity
            slot2 = sys_act.item[0].slot
            eType = sys_act.item[0].value_entity
            slot = sys_act.item[0].value
        else:
            logger.error('sys act does not address this entity.')
        
        if not goal.contains_slot_const(slot):
            # If slot is not in the goal, get the correct value for it.
            logger.warning('Slot %s in the given system act %s is not found in the user goal.' % (slot, str(sys_act)))
            #random_val = Ontology.global_ontology.getRandomValueForSlot(self.dstring, slot=slot)
            goal.add_const(self.dstring, slot, 'dontcare') # ic340 shouldnt this be dontcare instead of adding a random constrain? (or at least do it with a prob) su259: I agree and have changed it to dontcare
            self.agenda.push(DActEntity.DiaActEntity('inform(%s="%s")' % (slot, 'dontcare'), self.dstring))
        else:
            correct_val = goal.get_correct_const_value(slot, eType = eType)
            correct_rel = goal.get_correct_const_rel(slot, eType = eType)
            
            if correct_rel is not None: # preference is given for relation as this is what has been addressed
                dact = DActEntity.DiaActEntity('inform()', self.dstring)
                dact.addItem(correct_rel)
            elif correct_val is not None:
                self.agenda.push(DActEntity.DiaActEntity('inform(%s="%s")' % (slot, correct_val), self.dstring))
            else:
                logger.error('There is neither a value nor a relation in the goal which is odd as this has been checked earlier.')
            return

    def _receive_request_rel(self, sys_act, goal):
        pass # not implemented on system side yet
    
    
    def _receive_inform(self, sys_act, goal):
        # Check if the given inform act contains name=none.
        # If so, se the flag RELAX_CONSTRAINTS.
        possible_venue = []
        contains_name_none = False

        logger.debug('Received an inform act. Check if it contains name=none.')
        for item in sys_act.items:
            if item.slot == 'name':
                if item.op == '=' and item.val == 'none':
                    contains_name_none = True
                    self.relax_constraints = True
                    logger.debug('Yes it does. Try to correct or relax the given constraints.')
                elif item.op == '!=':
                    possible_venue.append(item.val)
                else:
                    self.relax_constraints = False

        # Reset requested slots right after the system recommend new venue.
        for item in sys_act.items:
            if item.slot == 'name' and item.op == '=' and item.val != 'none':
                if goal.requests['name'] != item.val:
                    goal.reset_requests()
                    break

        # Check the implicitly confirmed information.
        impl_confirm_ok = self._receive_implicit_confirm(sys_act, goal, False)
        if not impl_confirm_ok:
            logger.debug('The impl confirmed inform was not ok, so stop processing inform act.')
            return

        # If we get this far then all implicitly confirmed constraints were correctly understood.
        # If they don't match an item in the database, however, say bye or try again from beginning.
        sel_venue = None
        
        if not contains_name_none:
            self.informState[self.dstring].informState = True
        else:
            self.informState[self.dstring].informState = False
        
        if self.use_new_goal_scenarios:
            change_goal = False
            add_name_in_consts = False

            if contains_name_none:
                logger.debug('Definitely change the goal if there is no venue matching the current constraints.')
                change_goal = True
            elif self.first_venue_recommendation:
                self.first_venue_recommendation = False

                # Make a random choice of asking for alternatives,
                # even if the system has recommended another venue.
                f = Settings.random.rand()
                if f < self.rand_decision_probs['ReqAltsAfterVenRec1']:
                    # Ask for alternatives without changing the goal but add a !name in constraints.

                    # Insert name!=venue constraint.
                    goal.add_name_constraint(sys_act.get_value('name'), negate=True)

                    self.agenda.push(DiaAct.DiaAct('reqalts()'))
                    return

                elif f < self.rand_decision_probs['ReqAltsAfterVenRec1'] + self.rand_decision_probs['ReqAltsAfterVenRec2']:
                    # Do change the goal and ask for alternatives.
                    change_goal = True

                else:
                    # Decide not to ask for alternatives nor change the goal at this point.
                    goal.add_name_constraint(sys_act.get_value('name'))
            else:
                # After first venue recommendation we can't ask for alternatives again.
                goal.add_name_constraint(sys_act.get_value('name'))

            if change_goal:
                # Changing the goal.
                if len(goal.constraints) == 0:
                    logger.warning('No constraints available to change.')
                    change_goal = False
                else:
                    # Collect the constraints mentioned by the system act.
                    relax_candidates = []
                    for item in sys_act.items:
                        # Remember relax candidate that has to be set to dontcare.
                        set_dontcare= False
                        if contains_name_none and item.val == 'dontcare' and item.op == '!=':
                            set_dontcare = True
                        # Update candidate list
                        if item.slot not in ['name', 'type'] and\
                                Ontology.global_ontology.is_system_requestable(self.dstring, slot=item.slot) and\
                                item.val not in [None, goal.request_type] and\
                                (item.val != 'dontcare' or item.op == '!='):
                            relax_candidates.append((item.slot, set_dontcare))

                    # Pick a constraint to relax.
                    relax_dontcare = False
                    if len(relax_candidates) > 0:
                        index = Settings.random.randint(len(relax_candidates))
                        (relax_slot, relax_dontcare) = relax_candidates[index]
                    # Randomly pick a goal constraint to relax
                    else:
                        index = Settings.random.randint(len(goal.constraints))
                        relax_const = goal.constraints[index]
                        relax_slot = relax_const.slot

                    # Randomly decide whether to change it to another value or set it to 'dontcare'.
                    if relax_slot is not None:
                        #if type(relax_slot) not in [unicode, str]:
                        #    print relax_slot
                        #    logger.error('Invalid relax_slot type: %s in %s' % (type(relax_slot), relax_slot))
                        logger.debug('Relaxing constraint: ' + relax_slot)
                        if goal.contains_slot_const('name'):
                            goal.remove_slot_const('name')

                        # DEBUG THIS SECCTION:
                        logger.debug("choosen slot to relax constraints in: "+relax_slot)
                        logger.debug("current goal: "+str(goal))
                        logger.debug("current goal.prev_slot_values: "+str(goal.prev_slot_values))

                        if Settings.random.rand() < self.rand_decision_probs['ConstraintRelax'] or relax_dontcare:
                            logger.debug("--case0--")
                            # Just set it to dontcare.
                            relax_value = 'dontcare'
                        elif relax_slot not in goal.prev_slot_values:
                            logger.debug("--case1--")
                            # TODO - check this - added this elif as all domains bar CamRestaurants were crashing here
                            relax_value = 'dontcare'
                            goal.add_prev_used(relax_slot, relax_value) # is this necessary?
                        else:
                            logger.debug("--case2--")
                            # Set it to a valid value for this slot that is different from the previous one.
                            relax_value = Ontology.global_ontology.getRandomValueForSlot(self.dstring,
                                                                                        slot=relax_slot, 
                                                                                        nodontcare=True,
                                                                                        notthese=goal.prev_slot_values[relax_slot])
                            goal.add_prev_used(relax_slot, relax_value)
                            logger.debug("relax value to "+relax_value)

                        goal.replace_const(relax_slot, relax_value, eType = self.dstring)

                        # Randomly decide whether to tell the system about the change or just request an alternative.
                        if not contains_name_none:
                            if Settings.random.rand() < self.rand_decision_probs['TellAboutChange']:
                                # Decide to tell the system about it.
                                self.agenda.push(DiaAct.DiaAct('reqalts(%s="%s")' % (relax_slot, relax_value)))
                            else:
                                # Decide not to tell the system about it.
                                # If the new goal constraint value is set to something other than dontcare,
                                # then add the slot to the list of requests, so that the user asks about it
                                # at some point in the dialogue.
                                # If it is set to dontcare, add name!=value into constraint set.
                                self.agenda.push(DiaAct.DiaAct('reqalts()'))
                                if relax_value == 'dontcare':
                                    goal.add_name_constraint(sys_act.get_value('name'), negate=True)
                                else:
                                    goal.requests[relax_slot] = None
                        else:
                            # After inform(name=none,...) always tell the system about the goal change.
                            self.agenda.push(DiaAct.DiaAct('reqalts(%s="%s")' % (relax_slot, relax_value)))
                        return

                    else:
                        # No constraint to relax.
                        change_goal = False

            else: # change_goal == False
                # If name=none, ..., name!=x, ...
                if len(possible_venue) > 0:
                    # If # of possible venues is same to the number of name!=value constraints,
                    # that is, all possible venues are excluded by name!=value constraints.
                    # The user must relax them.
                    is_there_possible_venue = False
                    for venue in possible_venue:
                        if goal.is_satisfy_all_consts(dact.DactItem('name', '=', venue)):
                            is_there_possible_venue = True

                    if not is_there_possible_venue:
                        goal.remove_slot_const('name')

                    # Remove possible venues violating name constraints.
                    copy_possible_venue = copy.copy(possible_venue)
                    for venue in copy_possible_venue:
                        if not goal.is_satisfy_all_consts(dact.DactItem('name', '=', venue)):
                            possible_venue.remove(venue)

                    # 1) Choose venue from possible_venue, which satisfy the constraints.
                    sel_venue = Settings.random.choice(possible_venue)

                    # 2) Relax appropriate constraint from goal.
                    for cslot in copy.deepcopy(goal.constraints):
                        if not sys_act.contains_slot(cslot.slot):
                            # Constraint not found in system act: relax it.
                            goal.replace_const(cslot.slot, 'dontcare')

                            # Also remove any informs about this constraint from the agenda.
                            self.agenda.filter_acts_slot(cslot.slot)


        # Endif self.user_new_goal_scenarios == True
        if self.relax_constraints:
            # The given constraints were understood correctly but did not match a venue.
            if Settings.random.rand() < self.rand_decision_probs['ByeOrStartOver']:
                self.agenda.clear()
                self.agenda.push(DiaAct.DiaAct('bye()'))
            else:
                #self.agenda.push(DiaAct.DiaAct('inform(type=restaurant)'))
                self.agenda.push(DiaAct.DiaAct('inform(type=%s)' % goal.request_type ))
            return

        '''
        If we get this far then all implicitly confirmed constraints are correct.
        Use the given information to fill goal request slots.
        '''
        for slot in goal.requests:
            if slot == 'name' and sel_venue is not None:
                goal.requests[slot] = sel_venue
            else:
                for item in sys_act.items:
                    if item.slot == slot:
                        if item.op != '!=':
                            goal.requests[slot] = item.val
                        #if item.slot == 'name' and goal.nturns_to_first_recommendation == -1:
                        #    goal.nturns_to_first_recommendation = goal.nturns

        '''
        With some probability, change any remaining inform acts on the agenda to confirm acts.
        '''
        if Settings.random.rand() < self.rand_decision_probs['InformToConfirm']:
            for agenda_item in self.agenda.agenda_items:
                if agenda_item.act == 'inform':
                    if len(agenda_item.items) == 0:
                        logger.error('Empty inform act found on agenda.')
                    elif agenda_item.items[0].val != 'dontcare':
                        agenda_item.act = 'confirm'

        # Randomly decide to respond with thankyou() or ack(), or continue.
        if self.use_new_goal_scenarios:
            f = Settings.random.rand()
            if f < self.rand_decision_probs['ThankAck1']:
                self.agenda.push(DiaAct.DiaAct('thankyou()'))
                return
            elif f < self.rand_decision_probs['ThankAck1'] + self.rand_decision_probs['ThankAck2']:
                self.agenda.push(DiaAct.DiaAct('ack()'))
                return

        '''
        If empty slots remain in the goal, put a corresponding request for the first empty slot onto the agenda.
        If there are still pending acts on the agenda apart from bye(), though, then process those first,
        at least sometimes.
        '''
        if self.agenda.size() > 1:
            if Settings.random.rand() < self.rand_decision_probs['DealWithPending']:
                return

        # If empty goal slots remain, put a request on the agenda.
        if not goal.are_all_requests_filled():
            # Specify name in case the system giving complete list of venues matching constraints
            # inform(name=none, ..., name!=x, name!=y, ...)
            user_response = DiaAct.DiaAct('request()')
            if sel_venue is not None:
                # If user picked venue from multiple venues, it needs to specify selected name in request act.
                # If only one possible venue was offered, it specifies the name with some probability.
                user_response.append('name', sel_venue)

            one_added = False
            for slot in goal.requests:
                value = goal.requests[slot]
                if value is None:
                    if not one_added:
                        user_response.append(slot, None)
                        one_added = True
                    else:
                        # Add another request with some probability
                        if Settings.random.rand() < self.rand_decision_probs['AddSlotToReq']:
                            user_response.append(slot, None)
                        else:
                            break

            self.agenda.push(user_response)

    
    def _receive_select(self, sys_act, goal):
        # Receive select(slot=x, slot=y)
        slot = sys_act.items[0].slot
        value = sys_act.items[0].val
        if slot == 'name':
            self.agenda.push(DiaAct.DiaAct('inform(%s="%s")' % (slot, value)))
            # logger.error('select on name slot.')

        if not goal.contains_slot_const(slot):
            # If slot is not in the goal, get the correct value for it.
            logger.warning('Slot %s in the given system act %s is not found in the user goal.' % (slot, str(sys_act)))
            #random_val = Ontology.global_ontology.getRandomValueForSlot(self.dstring, slot=slot)
            goal.add_const(self.dstring, slot, 'dontcare') # ic340 shouldnt this be dontcare instead of adding a random constrain? (or at least do it with a prob) su259: I agree and have changed it to dontcare
            self.agenda.push(DiaAct.DiaAct('inform(%s="%s")' % (slot, 'dontcare')))
        else:
            correct_val = goal.get_correct_const_value(slot)
            self.agenda.push(DiaAct.DiaAct('inform(%s="%s")' % (slot, correct_val)))
            return

    def _normalise_act_no_rules(self, dap):
        #logger.debug(str(dap))
        norm_act = copy.deepcopy(dap)
        norm_act.items = []

        for item in dap.items:
            keep_it = True
            val = item.val
            slot = item.slot

            if slot == 'task':
                keep_it = False
            elif dap.act == 'request' and val is None:
                if slot == 'name':
                    keep_it = False
                    if val is None:
                        norm_act.act = 'inform'
                elif slot == 'bar' or slot == 'restaurant' or slot == 'hotel':
                    norm_act.append('type', slot)
                    keep_it = False
            elif slot is None and val is not None and val != 'dontcare':
                keep_it = False
                norm_act.append('type', val)

            if keep_it:
                if isinstance(norm_act, DActEntity.DiaActEntity):
                    norm_act.addItem(item)
                else:
                    norm_act.append(slot, val)

        #logger.debug(str(norm_act))
        return norm_act
    
    def _init_agenda(self):
        if self.informState is None:      
#             print "informState is None"     
            self.informState = {}
            
#         if self.informState == {}:
#             print "informState is empty"
            
        for o in self.informState:
            p = 0.0
            if self.informState[o].informState and Settings.config.has_option('usermodel', 'relationprob'):
                p = Settings.config.getfloat('usermodel', 'relationprob')
            self.relProb[o] = p
            logger.info("Setting relProb for {} to {} while being in {}: {}".format(o,p,self.dstring,self.informState))
            
        self.agenda.init(self.goal, self.relProb)
    
    def _checkIfRelStillHolds(self):
        # for each relation in current goal:
        relationList, _ = self.goal.get_relation_split()
        for item in relationList:
            slot = item.slot
        #     find value part of goal for that slot
            value = self.goal.get_correct_const_value(slot)
        #     find value part of related goal for that slot
            otherEntity = item.get_other_entity(self.dstring)
            otherValue = None
            if otherEntity in self.informState and self.informState[otherEntity] is not None:
                otherValue = self.informState[otherEntity].goal.get_correct_const_value(slot)
            
        #     if both are different: remove relation from goal
            if value != otherValue:
#                 print 'Relation {} removed from agenda due to value mismatch: self {}, {} {}'.format(item,value,otherEntity,otherValue)
#                 print self.goal
                logger.warning('Relation {} removed from agenda due to value mismatch: self {}, {} {}'.format(item,value,otherEntity,otherValue))
                self.goal.remove_slot_const(slot, eType=self.dstring, relOnly=True)
#                 print self.goal
        
    
class UMAgendaER(UserModel.UMAgenda):
    
    def __init__(self, dstring):
        super(UMAgendaER, self).__init__(dstring)
        self.initialized = False
    
    def init(self, goal, relProb):
        """
        Initialises the agenda by creating DiaActs corresponding to the
        constraints in the goal G. Uses the default order for the
        dialogue acts on the agenda: an inform act is created for
        each constraint. Finally a bye act is added at the bottom of the agenda.
        
        :param goal: 
               
        .. Note::
            No requests are added to the agenda.
        """
        self.agenda_items = []
        self.append_dact_to_front(DActEntity.DiaActEntity('inform(type="%s")' % goal.request_type, self.dstring))
        
        relationsList, nonrelationsList = goal.get_relation_split()
        
        slotsInAgenda = set()
        
        # first process relation List
        for const in relationsList:
            if Settings.random.rand() < relProb.get(self._getOtherObject(const), 0.0):
#                 print "Rel put on agenda {}".format(self.dstring)
                dia_act = DActEntity.DiaActEntity('inform()', self.dstring)
                dia_act.addItem(const)
                slotsInAgenda.add(const.slot)

        for const in nonrelationsList:
            slot = const.slot
            value = const.val
            if slot == 'method' or slot in slotsInAgenda:
                continue

            dia_act = DActEntity.DiaActEntity('inform()', self.dstring)
            dia_act.append(slot, value)
            if not self.contains(slot, value):
                self.append_dact_to_front(dia_act)

        # Finally append a bye() act to complete agenda:
        self.append_dact_to_front(DActEntity.DiaActEntity('bye()', self.dstring))
        
        self.initialized = True
        return
    
    
    def _getOtherObject(self, rel):
        if rel.slot_entity == self.dstring:
            return rel.value_entity
        else:
            return rel.slot_entity
        
class DomainUserState(object):
    
    def __init__(self):
        self.informState = False
        self.goal = None
        
    def __str__(self):
        return "informState {}".format(self.informState,self.goal)
    
    def __repr__(self, *args, **kwargs):
        return self.__str__()
    
    def reset(self, goal = None):
        self.informState = False
        self.goal = goal

#END OF FILE
