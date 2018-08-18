###############################################################################
# CUED PyDial: Multi-domain Statistical Spoken Dialogue System Software
###############################################################################
#
# Copyright 2015, 2016
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
RuleTopicTrackers.py - Rule based topic trackers 
==========================================================================

Copyright CUED Dialogue Systems Group 2015, 2016

.. seealso:: CUED Imports/Dependencies: 

    import :class:`utils.Settings` |.|
    import :class:`utils.ContextLogger` 

************************

'''
__author__ = "cued_dialogue_systems_group"

'''
    Modifications History
    ===============================
    Date        Author  Description
    ===============================
    Jul 20 2016 lmr46   Inferring only the domains configured in the config-file
                        Note that keywords for domains are set in the dictionary here (handcoded)
                        TODO: What happen when the same keyword apply for different domains?
'''
from utils import ContextLogger, Settings
from topictracking.RuleTopicTrackers import TopicTrackerInterface
logger = ContextLogger.getLogger('')

class DefaultValueDomainTracker(TopicTrackerInterface):
    """Template for any Topic Tracker for the cued-python system
    
    .. Note:
            To dynamically load a class, the __init__() must take one argument: domainString.
    """
    
    def __init__(self):
        self.defaultDomain = "default"
        
        if Settings.config.has_option("topictracker","defaultdomain"):
            self.defaultDomain = Settings.config.get("topictracker","defaultdomain")
        
    def infer_domain(self,userActHyps=None):
        return self.defaultDomain
    
    def restart(self):
        pass  # Define in actual class. May be some notion of state etc to be reset in more advanced topic trackers'''

