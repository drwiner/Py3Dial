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
DialogueState.py - dialogue state object specification
===========================================================

Copyright CUED Dialogue Systems Group 2017

**Basic Usage**: 
    >>> import DialogueState   
   
.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.ContextLogger` |.|

************************

'''

from copy import copy, deepcopy
from utils.ContextLogger import ContextLogger
#import pprint as pp
import utils.DialogueState
from ontology import Ontology
# DialogueState.DialogueState
logger = ContextLogger()


class DialogueState(utils.DialogueState.DialogueState):
    
    def printUserActs(self, dstring):
        '''
        Utility function to print the user acts stored in the belief state of domain dstring.
        
        :param dstring: the string identifier of the domain of which the user act should be printed 
        :type dstring: str
        '''
        if self.domainStates is not None and dstring in self.domainStates:
            self.domainStates[dstring].printUserActs()
            
    def check_user_ending(self):
        '''
        Utility function to check whether the user has said good bye.
        
        '''
        for domain in self.domainStates:
            if self.domainStates[domain].check_user_ending():
                return True
        return False
    
    def setLastSystemAct(self, sysAct):
        '''
        Sets the last system act of the current domain. Note that currentdomain needs to be set first, otherwise it does not work.
        
        :param sysAct: string representation of the last system action
        :type sysAct: str
        '''
        super(DialogueState, self).setLastSystemAct(sysAct)
        if self.currentdomain is not None:
            self.domainStates[self.currentdomain].setLastSystemAct(sysAct)
        else:
            logger.error('Attempt to store last system action for unknown domain.')
            
    def getLastSystemAct(self, dstring):
        '''
        Retreives the last system act of domain dstring.
        
        :param dstring: the string identifier of the domain of which the last system act should be retreived from
        :type dstring: str
        
        :returns: the last system act of domain dstring or None
        '''
        if dstring in self.lastSystemAct:
            return self.lastSystemAct[dstring]
        else:
            return None

    
class DomainDialogueState(object):
    '''
    classdocs
    '''
    
    def getDomainState(self, domain):
        if domain in self._entities:
            return self._entities[domain].belief
        return None


    def __init__(self):
        
        
        # several levels of state:
        # world: assume that we are always in the same world
        
        # world state:
        self._world = World() # handle bye discourse acts
        self.entityFocus = None
        
        # entity states:
        self._entities = {}
        self._relations = {}
        
        self.lastSystemAct = None
        
    def printUserActs(self, dstring = None):
        if dstring is None and 'userActs' in self._world.belief:
            print '   Usr > {}'.format(self.self._world.belief['userActs'])          
        
        elif self._entities is not None and 'userActs' in self._entities[dstring].belief:
            print '   Usr > {}'.format(self.domainStates[dstring].belief['userActs'])
                   
    def check_user_ending(self):
        if len(self._world.belief["beliefs"]):
            if 'bye' in self._world.belief["beliefs"]["discourseAct"]:
                if self._world.belief["beliefs"]["discourseAct"]['bye'] > 0.8:
                    return True             
        return False
            
    def getEntities(self):
        return self._entities.keys()
    
    def addEntity(self, e1):
        self._entities[e1] = Entity(e1)
        
    def getRelations(self):
        return self._relations.keys()
        
    def addRelation(self, e1, e2):
        rel_id = e1 + '#' + e2
        rel_id2 = e2 + '#' + e1
            
        if rel_id2 in self.getRelations():
            rel_id = rel_id2
        
        if rel_id not in self.getRelations():
            self._relations[rel_id] = Relation(e1, e2)
        
        return rel_id
        
    def getFocusBelief(self, newOne = False, getConflict = True):
        if self.entityFocus is not None and self.entityFocus in self._entities:
            if newOne:
                return self._computeFocusBeliefNew(self._entities[self.entityFocus], getConflict)
            else:
                return self._computeFocusBelief(self._entities[self.entityFocus])
        else:
            return None
        
    def getMergedBelief(self, entityId, newOne = False, getConflict = True):
        if newOne:
            belief = self._computeFocusBeliefNew(self._entities[entityId], getConflict)
        else:
            belief = self._computeFocusBelief(self._entities[entityId])
        if 'beliefs' in belief:
            slots = copy(belief['beliefs'].keys())
            for slot in slots:
                if "_" in slot:
                    del belief['beliefs'][slot]
            if 'requested' in belief['beliefs']:
                slots = copy(belief['beliefs']['requested'].keys())
                for slot in slots:
                    if '_' in slot:
                        del belief['beliefs']['requested'][slot]
        return belief
    
    def getExtendedRelationBeliefs(self, entityId):
        relations1, relations2 = self._findRelevantRelations(self._entities[entityId])
        
        beliefs = {}
        for rel in set(relations1 + relations2):
            beliefs[rel.eType] = rel.belief
            
        return beliefs
        
    def _relationIndicatesConflict(self, entity):
        probThreshold = 0.5
        relations1, relations2 = self._findRelevantRelations(entity)      
        
        conflict = False
        
        for rel in set(relations1 + relations2):
            for slot in rel.belief['beliefs']:
                if '=' not in rel.belief['beliefs'][slot] or not rel.belief['beliefs'][slot]['='] >= probThreshold:
                    continue
            
                if rel in relations1:
                    otherEntity = self._entities[rel.e2]
                elif rel in relations2:
                    otherEntity = self._entities[rel.e1]
                    
                s = slot.split('#')[0]
                if s in entity.belief['beliefs']:
                    maxP, maxVs = self._getMaxProbOfSlot(entity.belief['beliefs'][s])
                    if maxP >= probThreshold and s in otherEntity.belief['beliefs']:
                        maxOtherP, maxOtherVs = self._getMaxProbOfSlot(otherEntity.belief['beliefs'][s])
                        if maxOtherP >= probThreshold:
                            if set(maxVs) != set(maxOtherVs):
                                conflict = True
        return conflict
                            
              
        # for each relation of object enitiyId:
        #    check if relation is geq threshold of 0.5
        #    if yes, for each slot get max from object entityId and check if it is neq **NONE** and geq threshold of 0.5
        #    if yes, get max for same slot and value from other object and check if it is neq **NONE** and threshold geq 0.5
        #    if yes, return True
        # return False
        
        pass
     
    def _getMaxProbOfSlot(self,slotdict, excludeNone=True):
        maxProb = float('-inf')
        maxVals = []
        
        for val in slotdict:
            if val == '**NONE**':
                continue
            if slotdict[val] > maxProb:
                maxVals = [val]
                maxProb = slotdict[val]
            elif slotdict[val] == maxProb:
                maxVals.append(val)
                
        return maxProb, maxVals
            
        
    def _computeFocusBelief(self, entity):
        relations1, relations2 = self._findRelevantRelations(entity)
        
        belief = deepcopy(entity.getMergedBelief())
        
        rel_slots = Ontology.global_ontology.get_common_slots_for_all_types(entity.eType)
        
        
        # first construct initial relation belief which will be overwritten in case relation exists in state
        for domain in rel_slots:
            for slot in rel_slots[domain]:
                inform_slot_vals = ['=']
                belief['beliefs'][domain+'_'+slot] = dict.fromkeys(inform_slot_vals+['dontcare'], 0.0)
                belief['beliefs'][domain+'_'+slot]['**NONE**'] = 1.0
                belief['beliefs']['requested'].update({domain+'_'+slot : 0.0})
            
        
        
        # if no relations present, what do we do? 
        if not any(relations1) and not any(relations2):
            return belief
        
        
        # beliefs contains the merged belief information with the origin entity as dict key
        beliefs = {}
        for rel in relations1:
            if rel.e2 not in beliefs:
                # the beliefs entry of the specific entity origin is initialized with the belief state of the current focus entity
                beliefs[rel.e2] = deepcopy(entity.getMergedBelief())
            for slot in rel.belief['beliefs']:
                if 'request' in slot:
                    for req in rel.belief['beliefs']['requested']:
                        # things that can be copied are directly copied into final belief
                        belief['beliefs']['requested'][rel.e2+'_'+req] = deepcopy(rel.belief['beliefs']['requested'][req])
                elif 'discourseAct' in slot:
                    pass # sholud not happen
                elif 'method' in slot:
                    pass # sholud not happen
                else:
                    # first copy rel belief to current belief
                    # things that can be copied are directly copied into final belief
                    belief['beliefs'][rel.e2+'_'+slot] = deepcopy(rel.belief['beliefs'][slot])
                    
                    # second update actual entity slot belief by merging e2 with e1 using relation
                    s = slot.split('#')[0]
                    otherBelief = self._entities[rel.e2].getContextBelief()
                    otherBelief = self._copyAndEnsureSlotExists(otherBelief,s,beliefs[rel.e2]['beliefs'][s])
                    
                    for val in beliefs[rel.e2]['beliefs'][s]:
                        # special treatment of **NONE** and dontcare? => NO
                        # b_focus(v) = b(v) + b(none) * rel * (b'(v) - b(v))
                        try:
                            beliefs[rel.e2]['beliefs'][s][val] = belief['beliefs'][s][val] + ( belief['beliefs'][s]['**NONE**'] * rel.belief['beliefs'][slot]['='] * (otherBelief['beliefs'][s][val] - belief['beliefs'][s][val]) )
                        except KeyError as e:
                            print "beliefs[rel.e2]['beliefs']", beliefs[rel.e2]['beliefs']
                            print "belief['beliefs']", belief['beliefs']
                            print "rel.belief['beliefs']", rel.belief['beliefs']
                            print "otherBelief['beliefs']", otherBelief['beliefs']
                            raise e
        
        for rel in relations2:
            if rel.e1 not in beliefs:
                # the beliefs entry of the specific entity origin is initialized with the belief state of the current focus entity
                beliefs[rel.e1] = deepcopy(entity.getMergedBelief())
            for slot in rel.belief['beliefs']:
                if 'request' in slot:
                    for req in rel.belief['beliefs']['requested']:
                        # things that can be copied are directly copied into final belief
                        belief['beliefs']['requested'][rel.e1+'_'+req] = deepcopy(rel.belief['beliefs']['requested'][req])
                elif 'discourseAct' in slot:
                    pass # sholud not happen
                elif 'method' in slot:
                    pass # sholud not happen
                else:
                    # first copy rel belief to current belief
                    # things that can be copied are directly copied into final belief
                    belief['beliefs'][rel.e1+'_'+slot] = deepcopy(rel.belief['beliefs'][slot])
                    
                    # second update actual entity slot belief by merging e2 with e1 using relation
                    s = slot.split('#')[0]
                    otherBelief = self._entities[rel.e1].getContextBelief()
                    otherBelief = self._copyAndEnsureSlotExists(otherBelief,s,beliefs[rel.e1]['beliefs'][s])
                    
                    for val in beliefs[rel.e1]['beliefs'][s]:
                        # special treatment of **NONE** and dontcare? => NO
                        # b_focus(v) = b(v) + b(none) * rel * (b'(v) - b(v))
                        beliefs[rel.e1]['beliefs'][s][val] = belief['beliefs'][s][val] + ( belief['beliefs'][s]['**NONE**'] * rel.belief['beliefs'][slot]['='] * (otherBelief['beliefs'][s][val] - belief['beliefs'][s][val]) ) 
                
                
        # for now assume that there is only one relation because there are only two domains
        if any(beliefs):
            # merge beliefs using arithmetic average over all beliefs resulting from the different relations
            for slot in entity.getMergedBelief()['beliefs']:
                if slot == 'requested' or slot =='discourseAct' or slot == 'method':
                    continue
                for val in belief['beliefs'][slot]:
                    belief['beliefs'][slot][val] = 0.0
                for e in beliefs:
                    for val in belief['beliefs'][slot]:
                        belief['beliefs'][slot][val] += beliefs[e]['beliefs'][slot][val]
                for val in belief['beliefs'][slot]:
                    belief['beliefs'][slot][val] /= len(beliefs)
        
        return belief
    
    def _computeFocusBeliefNew(self, entity, getConflict = True):
        relations1, relations2 = self._findRelevantRelations(entity)
        
        belief = deepcopy(entity.belief)
        
        rel_slots = Ontology.global_ontology.get_common_slots_for_all_types(entity.eType)
        
        
        # first construct initial relation belief which will be overwritten in case relation exists in state
        for domain in rel_slots:
            for slot in rel_slots[domain]:
                inform_slot_vals = ['=']
                belief['beliefs'][domain+'_'+slot] = dict.fromkeys(inform_slot_vals+['dontcare'], 0.0)
                belief['beliefs'][domain+'_'+slot]['**NONE**'] = 1.0
                belief['beliefs']['requested'].update({domain+'_'+slot : 0.0})
        
        beliefs = {}
        for rel in set(relations1 + relations2):
            if rel in relations1:
                otherEntity = rel.e2
            elif rel in relations2:
                otherEntity = rel.e1
            
            beliefs[rel.eType] = {}
            for cat in rel.belief:
                if cat == 'beliefs':
                    beliefs[rel.eType]['beliefs'] = {}
                    for slot in rel.belief['beliefs']:
                        if 'request' in slot:
                            for req in rel.belief['beliefs']['requested']:
                                # things that can be copied are directly copied into final belief
                                belief['beliefs']['requested'][otherEntity+'_'+req] = deepcopy(rel.belief['beliefs']['requested'][req])
                        elif 'discourseAct' in slot or 'method' in slot:
                            # not needed
                            pass
#                             beliefs[rel][cat][slot] = deepcopy(rel.belief[cat][slot])
                        else:
                            # first copy rel belief to current belief
                            # things that can be copied are directly copied into final belief
                            belief['beliefs'][otherEntity+'_'+slot] = deepcopy(rel.belief['beliefs'][slot])
                            
                            
                            s = slot.split('#')[0]
                            otherBelief = self._entities[otherEntity].getContextBelief()
                            
                            weight = 1.0-rel.belief['beliefs'][slot]['**NONE**']
                            
                            if s in otherBelief['beliefs']:
                                for val in otherBelief['beliefs'][s]:
                                    if s not in beliefs[rel.eType][cat]:
                                        beliefs[rel.eType][cat][s] = dict()
                                    beliefs[rel.eType][cat][s][val] = weight * otherBelief['beliefs'][s][val]
                            else:
                                if s not in beliefs[rel.eType][cat]:
                                    beliefs[rel.eType][cat][s] = dict()
                                beliefs[rel.eType][cat][s]['**NONE**'] = weight * 1.0
                                
                            beliefs[rel.eType][cat][s]['**NONE**'] += rel.belief['beliefs'][slot]['**NONE**']
                else:
                    beliefs[rel.eType][cat] = deepcopy(rel.belief[cat])
                    
     
        for slot in entity.belief['beliefs']: # iterate only over original entity slots
            totalWeights = 0.0
            if 'request' in slot or 'discourseAct' in slot or 'method' in slot:
                # not needed
                pass
            else:
                weight = 1.0-belief['beliefs'][slot]['**NONE**']               
                for val in belief['beliefs'][slot]:
                    belief['beliefs'][slot][val] *= weight # should be normalized later again if no other belief mass is added
                totalWeights += weight
                
                for b in beliefs:
                    if slot in beliefs[b]['beliefs']:
                        weight = 1.0-beliefs[b]['beliefs'][slot]['**NONE**']
                        for val in belief['beliefs'][slot]:
                            try:
                                if val not in beliefs[b]['beliefs'][slot]:
                                    belief['beliefs'][slot][val] += weight * 0.0
                                else:
                                    belief['beliefs'][slot][val] += weight * beliefs[b]['beliefs'][slot][val]
                            except KeyError as e:
                                logger.error('Error: {}, val: {}, slot: {}, b: {}'.format(e,val,slot,b))
                        totalWeights += weight
        
                if totalWeights != 0.0:
                    for val in belief['beliefs'][slot]:
                        belief['beliefs'][slot][val] /= totalWeights
                else:
                    belief['beliefs'][slot]['**NONE**'] = 1.0 # should only happen if there is no evidence anywhere
        
        if getConflict:            
            belief['features']['relationConflict'] = self._relationIndicatesConflict(entity)
                    
        return belief
    
    def _copyAndEnsureSlotExists(self,otherBelief,slot,cmpBeliefOfSlot):
        
        otherBelief = deepcopy(otherBelief)
        if slot not in otherBelief['beliefs']:
            otherBelief['beliefs'][slot] = dict()
            for val in cmpBeliefOfSlot:
                otherBelief['beliefs'][slot][val] = 0.0
                otherBelief['beliefs'][slot]['**NONE**'] = 1.0
        
        return otherBelief
        
        
    def _findRelevantRelations(self, entity):
        relations1 = set() # relations where this entity is e1
        relations2 = set() # relations where this entity is e2
        for rel in self._relations:
            if self._relations[rel].e1 == entity.eType:
                relations1.add(self._relations[rel])
            if self._relations[rel].e2 == entity.eType:
                relations2.add(self._relations[rel])
        return list(relations1), list(relations2)
        
        
    def getLastUserActs(self):
        acts = [self._entities[e].lastHyps for e in self._entities if self._entities[e] is not None and self._entities[e].lastHyps is not None]
        acts.extend([self._relations[r].lastHyps for r in self._relations if self._relations[r] is not None and self._relations[r].lastHyps is not None])
        
        eol = False
        actList = []
        i = 0
        while not eol:
            l = [a[i] for a in acts if len(a) > i]
            if not len(l):
                eol = True
            else:
                actList.append(l) 
                i += 1
        
        return actList
        
    def setLastSystemAct(self, sysAct):
        self.lastSystemAct = sysAct
        if self._world is not None:
            self._world.lastSystemAct = sysAct
        if sysAct.entityname in self._entities:
            self._entities[sysAct.entityname].lastSystemAct = sysAct 
            self._entities[sysAct.entityname].updateContext(sysAct)
        
        # TODO set context
        
    def getEntity(self, e):
        return self._entities[e]
    
    #---- below not yet implemented
    
    def updateRelation(self, rel):
        pass
    
    def getRelation(self, e1, e2):
        pass
    
    def updateEntity(self, e):
        pass
    
    
        
class UniverseElement(object):
    
    def __init__(self):
        self.belief = {"beliefs":{}}
        self.lastHyps = {"goal-labels":{},"method-label":{}, "requested-slots":{}} # necessary for focus tracking
        self.lastSystemAct = None
        self.context = {}
        self.prior = {}
        
    def updateUserInput(self, obs):
        self.belief['userActs'] = obs
        
    def getUserInput(self):
        return self.belief['userActs']  
    
    def updateContext(self, sysAct):
        pass
    
    def getMergedBelief(self):
        # merging prior and belief
        return self.belief
    
    def getContextBelief(self):
        if self.context != {}:
            newBelief = deepcopy(self.belief)
            for slot in self.context:
                if slot in newBelief['beliefs']:
                    for value in newBelief['beliefs'][slot]:
                        newBelief['beliefs'][slot][value] = 0.0
                    newBelief['beliefs'][slot][self.context[slot]] = 1.0
            return newBelief
        return deepcopy(self.belief)
    
    def setPrior(self):
        pass
        
class Entity(UniverseElement):
    
    def __init__(self, t):
        super(Entity, self).__init__()
        self.eType = t
        
    def updateContext(self, sysAct):
        if 'inform' in sysAct.act:
            # check if it contains none (multiple entries possible)
            containsNameNone = False
            for item in sysAct.items:
                if item.slot == 'name' and item.op == '=' and item.val in ['none', 'NONE', '**NONE**']:
                    containsNameNone = True
                    
            if not containsNameNone:
                # check if name is none or something real
                for item in sysAct.items:
                    if item.slot == 'name' and item.op == '=':
                        if item.val not in ['none', 'NONE', '**NONE**']:
                            # 1) get entity from database
                            entities = Ontology.global_ontology.entity_by_features(self.eType, sysAct.items)
                            # 2) set context dict with respective slot/value pairs
                            if len(entities) > 1:
                                pass
                            elif len(entities) == 1:
                                self.context = deepcopy(entities[0])
                            else:
                                logger.error('No matching entity found, how is this possible?')
                    else:
                        # delete context when name=none
                        self.context = {}
            else:
                # delete context when name=none
                self.context = {}
                        
                    
    
class Relation(UniverseElement):
    
    def __init__(self, e1, e2):
        super(Relation, self).__init__()
        self.eType = e1+'#'+e2
        self.e1 = e1
        self.e2 = e2
        
        
class World(UniverseElement):   
    pass
        
    
        
        