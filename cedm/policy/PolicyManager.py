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
PolicyManager.py - container for all policies
===============================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger` |.|
    import :mod:`ontology.Ontology` |.|
    import :mod:`ontology.OntologyUtils`

************************

'''
__author__ = "cued_dialogue_systems_group"

#from utils import Settings
from utils import ContextLogger
from cedm.utils import DActEntity
#from ontology import Ontology,OntologyUtils
from policy import PolicyManager
logger = ContextLogger.getLogger('')

class PolicyManager(PolicyManager.PolicyManager):
        
    def act_on(self, dstring, state):
        '''
        Main policy method which maps the provided belief to the next system action. This is called at each turn by :class:`~Agent.DialogueAgent`
        
        :param dstring: the domain string unique identifier.
        :type dstring: str
        :param belief: the belief state the policy should act on
        :type belief: dict
        :returns: the next system action
        '''
        domainstate = state.domainStates[dstring]
        
        etype = domainstate.entityFocus
        
        if etype == dstring:
            domainstate = state

        if self.domainPolicies[etype] is None:
            self.bootup(etype)
                
        if self.committees[etype] is not None:
            _systemAct = self.committees[dstring].act_on(state=domainstate, domainInControl=etype)
        else:
            _systemAct = self.domainPolicies[etype].act_on(state=domainstate)
            
        systemAct = DActEntity.DiaActEntity(_systemAct, etype)
           
        return systemAct
    
    
    def record(self, reward, domainString):
        '''
        Records the current turn reward for the given domain. In case of a committee, the recording is delegated. 
        
        This method is called each turn by the :class:`~Agent.DialogueAgent`.
        
        :param reward: the turn reward to be recorded
        :type reward: int
        :param domainString: the domain string unique identifier of the domain the reward originates in
        :type domainString: str
        :returns: None
        '''        
        if self.committees[domainString] is not None:
            self.committees[domainString].record(reward, domainString)
        else:
            self.domainPolicies[domainString].record(reward)
            
    def finalizeRecord(self, domainRewards):
        '''
        Records the final rewards of all domains. In case of a committee, the recording is delegated. 
        
        This method is called once at the end of each dialogue by the :class:`~Agent.DialogueAgent`. (One dialogue may contain multiple domains.)
        
        :param domainRewards: a dictionary mapping from domains to final rewards
        :type domainRewards: dict
        :returns: None
        '''
        for dstring in self.domainPolicies:
            if self.domainPolicies[dstring] is not None:
                domains_reward = domainRewards[dstring]
                if domains_reward is not None:
                    if self.committees[dstring] is not None:
                        self.committees[dstring].finalizeRecord(domains_reward,dstring)
                    elif self.domainPolicies[dstring] is not None:
                        self.domainPolicies[dstring].finalizeRecord(domains_reward,dstring)
                else:
                    logger.warning("Final reward in domain: "+dstring+" is None - Should mean domain wasnt used in dialog")
                    
    
    
