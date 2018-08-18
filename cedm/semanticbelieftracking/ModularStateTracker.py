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
ModularSemanticBeliefTraker.py - separate modelling of semantic decoding and belief tracking
============================================================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.ContextLogger` |.|
    import :mod:`belieftracking.BeliefTrackingManager` |.|
    import :mod:`semi.SemI` |.|
    import :class:`semanticbelieftracking.SemanticBeliefTrackingManager.SemanticBeliefTracker`

*********************************************************************************************

'''

from utils import ContextLogger, Settings
from cedm.utils.Belief import DomainDialogueState
from semanticbelieftracking.SemanticBeliefTrackingManager import SemanticBeliefTracker
from semi import SemI
from belieftracking import BeliefTrackingManager

logger = ContextLogger.getLogger('')

    
class ModularStateTracker(SemanticBeliefTracker):
    '''
    This class implements the functionality of the original spoken dialogue systems pipeline where semantic decoding and belief tracking
    are looked at as two separate problems. Refers all requests to :class:`semi.SemI.SemIManager` and :class:`belieftracking.BeliefTrackingManager.BeliefTrackingManager`.
    '''
    
    belief_manager = None
    semi_manager = None
    
    def __init__(self, dstring):
#         super(ModularStateTracker, self).__init__(eType)
        
        self.semi_manager = ModularStateTracker.getSemiManager()
        self.belief_manager = ModularStateTracker.getBeliefTrackingManager()
        self.entityTracker = None
        self.domainString = dstring
        
        
#         semibelief_TownInfo
        
        entityTracker_type = None  # TopicTracker may not be needed in simulate, 
        if Settings.config.has_option("semibelief_"+self.domainString,"entitytracker"):
            entityTracker_type = Settings.config.get("semibelief_"+self.domainString,"entitytracker")
        
        if entityTracker_type is not None:
            try:
                # try to view the config string as a complete module path to the class to be instantiated
                components = entityTracker_type.split('.')
                packageString = '.'.join(components[:-1]) 
                classString = components[-1]
                mod = __import__(packageString, fromlist=[classString])
                klass = getattr(mod, classString)
                self.entityTracker = klass()
            except ImportError:
                logger.error('Unknown entity tracker "{}"'.format(entityTracker_type))
        else:
            logger.error("Entity tracker not specified in config.")
            
            
        semiManager_type = None  # TopicTracker may not be needed in simulate, 
        if Settings.config.has_option("semibelief_"+self.domainString,"semimanager"):
            semiManager_type = Settings.config.get("semibelief_"+self.domainString,"semimanager")
        if semiManager_type is not None:
            try:
                # try to view the config string as a complete module path to the class to be instantiated
                components = semiManager_type.split('.')
                packageString = '.'.join(components[:-1]) 
                classString = components[-1]
                mod = __import__(packageString, fromlist=[classString])
                klass = getattr(mod, classString)
                self.semi_manager = klass()
            except ImportError:
                logger.error('Unknown semi manager "{}"'.format(semiManager_type))
#         else:
#             logger.error("SemI manager not specified in config.")
    
        beliefManager_type = None  # TopicTracker may not be needed in simulate, 
        if Settings.config.has_option("semibelief_"+self.domainString,"beliefmanager"):
            beliefManager_type = Settings.config.get("semibelief_"+self.domainString,"beliefmanager")
        if beliefManager_type is not None:
            try:
                # try to view the config string as a complete module path to the class to be instantiated
                components = beliefManager_type.split('.')
                packageString = '.'.join(components[:-1]) 
                classString = components[-1]
                mod = __import__(packageString, fromlist=[classString])
                klass = getattr(mod, classString)
                self.belief_manager = klass()
            except ImportError:
                logger.error('Unknown belief manager "{}"'.format(beliefManager_type))
#         else:
#             logger.error("Belief manager not specified in config.")

        self.restart()
        
    
    def update_belief_state(self, ASR_obs, sys_act, constraints, turn=None, hub_id = None, sim_lvl = None):
        # current assumptions: 
        #     - only one relation addressed in user input if no relation
        #     - relations only between entities which have already been mentioned in the dialogue 
        
        
        
        if hub_id == 'simulate':
        # to update: word model, world belief, entity belief, focus
        
        # world model:
        #   assume that we operate in a predefined world with one entity per specified domain
        
        # world belief: b_w(u_w,e)
        
        ## semantically decode input wrt given world 
        ## for simulate => already done as input is on semantic level 
            if ASR_obs is None:
                ASR_obs = []
            self.state._world.updateUserInput(ASR_obs)
        
        ## pass input to general belief tracker only tracking non-goal information
            self.belief_manager.update_world_belief_state(self.state._world, self.state.lastSystemAct)
        
        # entity belief: b_e(u_e,s_u,s_c,h)
        
        ## figure out objects and relations in input and update lastHyps accordingly
            # objects part of input addressed without relation
            object_obs = []
            relation_obs = []
            objects = set() # list of single
            relations = set() # list of tuples
            
            if ASR_obs is not None and ASR_obs != []:
                separated_obs = [h.separate(self.state.lastSystemAct) for h in ASR_obs]
                object_obs = [o for (o, _) in separated_obs if o is not None]
                relation_obs = [r for (_, r) in separated_obs if r is not None]
                
                # FIND ENTITIES AND RELATIONS WHICH ARE ADDRESSED BY INPUT
                objects.update(set([o.entityname for o in object_obs]))

                items_in_relations = list()
                for relation_ob in relation_obs:
                    items_in_relations.extend(relation_ob.items)
                    relations.update(set([tuple(sorted(relation_ob.entityname.split('#')))]))
                relations.update(set([tuple(sorted([item.slot_entity, item.value_entity])) for item in items_in_relations]))
              
                
#                 o_obs = [(h.to_string_plain(), h.P_Au_O) for h in object_obs]
                for o in objects:
                    if o not in self.state._entities:
                        self.state.addEntity(o)                    
                    # only process entities which are part of input
                    self.state._entities[o].updateUserInput(self.semi_manager.simulate_add_context_to_user_act(self.state.lastSystemAct, object_obs, o))
                    
#                 r_obs = [(h, h.P_Au_O) for h in relation_obs]
                    
                for relation in relations:
                    
                    if relation[0] == relation[1]:
                        logger.error("Self-relation detected: {} {}".format(relation[0], relation[1]))
                    
                    # make sure all objects addressed are initialized (also objects which are only part of relations)
                    if relation[0] not in self.state._entities:
                        self.state.addEntity(relation[0])  
                    if relation[1] not in self.state._entities:
                        self.state.addEntity(relation[1])
                    
                    rel_id = self.state.addRelation(relation[0], relation[1]) # checks automatically if relation already exists
                    self.state._relations[rel_id].updateUserInput(self.semi_manager.simulate_add_context_to_user_act_relation(self.state.lastSystemAct, relation_obs, rel_id))
        
        ## update internal state of objects and relations found in input; order matters for DB-feature update in rule-based merging 
            for relation in relations:
                rel_id = self.state.addRelation(relation[0], relation[1]) # checks automatically if relation already exists
                self.belief_manager.update_belief_state_relation(self.state._relations[rel_id], self.state.lastSystemAct)
                
            for o in objects:
                # only process entities which are part of input
                self.belief_manager.update_belief_state(self.state._entities[o], self.state.lastSystemAct)
         
        ######################## checked until here, date: 06.11.2017 ###################
        
        # FIND ENTITY FOCUS (necessary for downstream modules)
        entityFocus = None
        if ASR_obs is not None and ASR_obs != []:
            entityFocus = self.entityTracker.track_topic(ASR_obs, ASR_obs[0].entityname)
        else:
            entityFocus = self.entityTracker.track_topic([('','')])
            
        # ENSURE ENTITIES IN FOCUS ARE INITIALISED -- currently only one at a time
        if entityFocus not in self.state.getEntities() and entityFocus != self.domainString:
            self.state.addEntity(entityFocus)
            
        self.state.entityFocus = entityFocus
            
#         # ER-TODO select focus based on last system action... relation or relation?
#         if self.state.entityFocus != self.domainString:
#             entities.add(self.state.entityFocus)

        
        return self.state
    
    def restart(self, previousDomainString = None):
#         super(ModularStateTracker,self).restart(previousDomainString)
        self.state = DomainDialogueState()
        self.entityTracker.restart()
        #self.semi_manager.restart()
        self.belief_manager.restart()
        return
    
#     def hand_control(self, previousDomain):
#         return # nothing to do here yet, will there ever be?
# #         return self.belief_manager.conditionally_init_new_domains_belief(self.domainString, previousDomain)
    
    @staticmethod
    def getBeliefTrackingManager():
        if ModularStateTracker.belief_manager is None:
            ModularStateTracker.belief_manager = BeliefTrackingManager.BeliefTrackingManager()
        return ModularStateTracker.belief_manager
    
    @staticmethod
    def getSemiManager():
        if ModularStateTracker.semi_manager is None:
            ModularStateTracker.semi_manager = SemI.SemIManager()
        return ModularStateTracker.semi_manager
        
