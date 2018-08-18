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
Created on 15 Dec 2017

@author: su259
'''

from utils import Settings, ContextLogger
from usersimulator import UserModel
logger = ContextLogger.getLogger('')

class UMCE(UserModel.UM):
    def init(self, otherDomainsConstraints, goal_with_rel):
        '''
        Initialises the simulated user.
        1. Initialises the goal G using the goal generator.
        2. Populates the agenda A using the goal G.
        Resets all UM status to their defaults.

        :param otherDomainsConstraints: of domain goals/constraints (slot=val) from other domains in dialog for which goal has already been generated.
        :type otherDomainsConstraints: list
        :returns None:
        '''
        if self.sampleParameters:
            self._sampleParameters()

        if self.sample_patience:
            self.max_patience = Settings.random.randint(self.sample_patience[0], self.sample_patience[1])
        
        self.goal = self.generator.init_goal(otherDomainsConstraints, self.max_patience, goal_with_rel)
        logger.debug(str(self.goal))

        self.lastUserAct = None
        self.lastSysAct = None
        self.hdcSim.init(self.goal, self.max_patience)  #uses infor in self.goal to do conditional generation of agenda as well.
        
        
        