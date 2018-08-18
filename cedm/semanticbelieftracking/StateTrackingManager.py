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
SemanticBeliefTrakingManager.py - module handling mapping from words to belief state
====================================================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger` |.|
    import :mod:`ontology.OntologyUtils`

************************

'''

from utils import ContextLogger
from cedm.utils.Belief import DialogueState
import semanticbelieftracking.SemanticBeliefTrackingManager
logger = ContextLogger.getLogger('')



class StateTrackingManager(semanticbelieftracking.SemanticBeliefTrackingManager.SemanticBeliefTrackingManager):
    '''
    The semantic belief tracking manager manages the semantic belief trackers for all domains. A semantic belief tracker is a mapping from words/ASR input to belief state.
    
    Its main interface method is :func:`update_belief_state` which updates and returns the internal belief state on given input.
    
    Internally, a dictionary is maintained which maps each domain to a :class:`SemanticBeliefTracker` object which handles the actual belief tracking.
    '''
    
    def __init__(self):
        super(StateTrackingManager, self).__init__()
        self.state = DialogueState()
    
    def restart(self):
        '''
        Restarts all semantic belief trackers of all domains and resets internal variables.
        '''
        super(StateTrackingManager, self).restart()
        self.state = DialogueState()
        return
    
   
