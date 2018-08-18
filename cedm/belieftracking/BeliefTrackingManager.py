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
BeliefTrackingManager.py - wrapper for belief tracking  for CEDM
======================================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.Settings` |.|
    import :mod:`ontology.OntologyUtils` |.|
    import :mod:`utils.ContextLogger` |.|
    import :class:`cedm.belieftracking.WorldBeliefTracker.WorldFocusTracker`

************************

'''


__author__ = "cued_dialogue_systems_group"
from utils import Settings
from ontology import OntologyUtils
from utils import ContextLogger
from WorldBeliefTracker import WorldFocusTracker
logger = ContextLogger.getLogger('') 



#--------------------------------------------------------------------------------------------------
# BELIEF MANAGER - controls each domains belief tracking abilities
#--------------------------------------------------------------------------------------------------
class BeliefTrackingManager(object):
    '''
    Higher-level belief tracker manager 
    '''
    def __init__(self):
        self.typeBeliefTrackers = dict.fromkeys(OntologyUtils.available_domains, None)
        self.worldBeliefTracker = WorldFocusTracker()
        self.SPECIAL_TYPES = ['topicmanager','wikipedia','ood']
        
        # TODO change to state.entity.prior
        
#         self.CONDITIONAL_BELIEF = False
#         if Settings.config.has_option("conditional","conditionalbeliefs"):
#             self.CONDITIONAL_BELIEF = Settings.config.getboolean("conditional","conditionalbeliefs")
#         self.prev_domain_constraints = None
    
    def restart(self):
        '''
        Restart every alive Belief Tracker
        '''
        for eType in self.typeBeliefTrackers.keys():
            if self.typeBeliefTrackers[eType] is not None: 
                self.typeBeliefTrackers[eType].restart()
                
        self.worldBeliefTracker.restart()
        
#         if self.CONDITIONAL_BELIEF:
#             self.prev_domain_constraints = dict.fromkeys(OntologyUtils.available_domains) # used to facilate cond. behaviour
    
    def update_belief_state(self, entity, lastSysAct):
        '''
        Update belief state given infos
        '''
        if self.typeBeliefTrackers[entity.eType] is None:
            self.bootup(entity.eType)
        return self.typeBeliefTrackers[entity.eType].update_belief_state(entity, lastSysAct)
    
    def update_belief_state_relation(self, relation, lastSysAct):
        '''
        Update belief state for relations given infos
        '''
        if relation.eType not in self.typeBeliefTrackers or self.typeBeliefTrackers[relation.eType] is None:
            self.bootup_rel(relation.eType)
        return self.typeBeliefTrackers[relation.eType].update_belief_state(relation, lastSysAct)
    
    def update_world_belief_state(self, world, lastSysAct):
        '''
        Update belief state for world given infos (currently only tracking discourse act)
        '''        
        return self.worldBeliefTracker.update_belief_state(world, lastSysAct)
        

    def bootup(self, eType):
        '''
        Boot up the belief tracker
        
        :param eType: domain name
        :type eType: string

        :param previousDomainString: previous domain name
        :type previousDomainString: string

        :return: None
        '''
        self._load_etype_belieftracker(eType)
#         return self.conditionally_init_new_domains_belief(eType, previousDomainString)
        return
    
    def bootup_rel(self, rType):
        '''
        Boot up the belief tracker
        
        :param eType: domain name
        :type eType: string

        :param previousDomainString: previous domain name
        :type previousDomainString: string

        :return: None
        '''
        import RelationBeliefTracker
        self.typeBeliefTrackers[rType] = RelationBeliefTracker.RelationFocusTracker(rType)
#         return self.conditionally_init_new_domains_belief(eType, previousDomainString)
        return
        

#     def conditionally_init_new_domains_belief(self, eType, origin_eType):
#         """
#         If just starting this domain in this dialog: Get count80 slot=value pairs from previous
#         domains in order to initialise the belief state of the new domain (reflecting dialogs history
#         and likelihood that similar values will be desired if there are slot overlaps.
# 
#         :param eType: domain name
#         :type eType: string
# 
#         :param origin_eType: previous domain name
#         :type origin_eType: string
# 
#         :return: None
#         """
#         if origin_eType in self.SPECIAL_TYPES:
#             return  # no information from these domains to carry over 
#         if origin_eType is not None and self.CONDITIONAL_BELIEF:
#             # 1. get 'count80' slot=values:
#             self.prev_domain_constraints[origin_eType] = self.typeBeliefTrackers[origin_eType].getBelief80_pairs()
#             # 2. initialise belief in (this dialogs) new domain:
#             return self.typeBeliefTrackers[eType].get_conditional_constraints(self.prev_domain_constraints)  
#         else:
#             return
        
    def _load_etype_belieftracker(self, eType=None):
        '''
        Load domain's belief tracker

        :param eType: domain name
        :type eType: string

        :return: None
        '''
        belief_type = 'focus'
        
        if Settings.config.has_option('policy_'+eType, 'belieftype'):
            belief_type = Settings.config.get('policy_'+eType, 'belieftype')

        if belief_type == 'focus':
            from ObjectBeliefTracker import ObjectFocusTracker
            self.typeBeliefTrackers[eType] = ObjectFocusTracker(eType)
        elif belief_type == 'baseline':
            from baseline import BaselineTracker
            self.typeBeliefTrackers[eType] = BaselineTracker(eType)
#         elif belief_type == 'rnn':
#             beliefs = BeliefTrackerRules.RNNTracker()
        else:
            try:
                # try to view the config string as a complete module path to the class to be instantiated
                components = belief_type.split('.')
                packageString = '.'.join(components[:-1]) 
                classString = components[-1]
                mod = __import__(packageString, fromlist=[classString])
                klass = getattr(mod, classString)
                self.typeBeliefTrackers[eType] = klass(eType)
            except ImportError:
                logger.error('Invalid semantic belief tracking type "{}" for entity type "{}"'.format(belief_type, eType))

#END OF FILE
