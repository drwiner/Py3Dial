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
GPPolicy.py - Gaussian Process policy 
============================================

Copyright CUED Dialogue Systems Group 2015 - 2017

   
**Relevant Config variables** [Default values]::

    [gppolicy]
    kernel = polysort
    thetafile = ''    

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`policy.GPLib` |.|
    import :mod:`policy.Policy` |.|
    import :mod:`policy.PolicyCommittee` |.|
    import :mod:`ontology.Ontology` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger`

************************

'''

__author__ = "cued_dialogue_systems_group"



import FeudalSubPolicy
import SummaryActionRel
from policy import SummaryUtils
from cedm.utils import DActEntity
from utils import ContextLogger
logger = ContextLogger.getLogger('')

class GPPolicy(FeudalSubPolicy.FeudalGPSubPolicy):
    '''
    An implementation of the dialogue policy based on Gaussian process and the GPSarsa algorithm to optimise actions where states are GPState and actions are GPAction.
    
    The class implements the public interfaces from :class:`~Policy.Policy` and :class:`~PolicyCommittee.CommitteeMember`.
    '''
    def __init__(self, domainString, learning, sharedParams=None):
        super(GPPolicy, self).__init__(domainString,learning,sharedParams)
        
        self.actions = SummaryActionRel.SummaryActionRel(domainString, False, self.useconfreq)
        # Total number of system actions.
        self.numActions = len(self.actions.action_names)
        
    def act_on(self, state):
        '''
        Main policy method: mapping of belief state to system action.
        
        This method is automatically invoked by the agent at each turn after tracking the belief state.
        
        May initially return 'hello()' as hardcoded action. Keeps track of last system action and last belief state.  
        
        :param state: the belief state to act on
        :type state: :class:`~utils.DialogueState.DialogueState`
        :param hyps: n-best-list of semantic interpretations
        :type hyps: list
        :returns: the next system action of type :class:`~utils.DiaAct.DiaAct`
        '''
        beliefstate = state.getFocusBelief( newOne = True, getConflict = True)
        beliefstate['features']['inform_info'] = self._updateDBfeatures(state.getMergedBelief(state.entityFocus),state.entityFocus)
        
        if self.lastSystemAction is None and self.startwithhello:
            _systemAct = 'hello()'
        else:
            _systemAct = self.nextAction(beliefstate)
        self.lastSystemAction = _systemAct
        self.prevbelief = beliefstate
        
        systemAct = DActEntity.DiaActEntity(_systemAct, state.entityFocus)
        return systemAct
    
    def _updateDBfeatures(self,belief,eType):
        features = []
        for numAccepted in range(1,6):
            temp =  SummaryUtils.actionSpecificInformSummary(belief, numAccepted, eType)
            features += temp
        return features
    
        
# END OF FILE
