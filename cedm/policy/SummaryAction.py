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
SummaryAction.py - Mapping between summary and master actions
=============================================================

Copyright CUED Dialogue Systems Group 2015 - 2017, 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`policy.SummaryUtils` |.|
    import :mod:`ontology.Ontology` |.|
    import :mod:`utils.ContextLogger` |.|
    import :mod:`utils.Settings`

************************

'''


__author__ = "cued_dialogue_systems_group"

import policy.SummaryAction
from utils import ContextLogger
logger = ContextLogger.getLogger('')
MAX_NUM_ACCEPTED = 10


class SummaryAction(policy.SummaryAction.SummaryAction):
    '''
    The summary action class encapsulates the functionality of a summary action along with the conversion from summary to master actions.
    
    .. Note::
        The list of all possible summary actions are defined in this class.
    '''
    def __init__(self, domainString, empty=False, confreq=False):
        '''
        Records what domain the class is instantiated for, and what actions are available

        :param domainString: domain tag
        :type domainString: string
        :param empty: None
        :type empty: bool
        :param confreq: representing if the action confreq is used
        :type confreq: bool
        '''
        super(SummaryAction, self).__init__(domainString, empty, confreq)
        
        self.action_names.append('pass')


    def getNonExecutable(self, belief, lastSystemAction):
        nonexec = super(SummaryAction, self).getNonExecutable(belief, lastSystemAction)
        nonexec.append('pass')

        logger.dial('masked inform actions:' + str([act for act in nonexec if 'inform' in act]))
        return nonexec
    
#END OF FILE
