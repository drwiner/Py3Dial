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
Created on 19 Oct 2016

@author: su259
'''
from utils import Settings, ContextLogger
logger = ContextLogger.getLogger('')

class WeightGenerator(object):
    '''
    classdocs
    '''


    def __init__(self, domainString, weightGeneration = None, distribution = None):
        self.domainString = domainString
        
        self.probs = []
        self.weightList = []
        total = 1

        if weightGeneration is None:
            weightGeneration = "auto" # auto or hdc
        if distribution is None:
            distribution = "uniform" # valley or uniform
        
        if weightGeneration == "auto":
            for i in range(101):
                self.weightList.append(float(i)/100.0)
        elif weightGeneration == "hdc":
            self.weightList = [0.1,0.3,0.5,0.7,0.9]
                

        if distribution == "uniform":
            self.probs = [1] * len(self.weightList)
            total = len(self.weightList)
        elif distribution == "valley":
            # works only for weightLists of uneven length
            value = len(self.weightList)-1
            middle = len(self.weightList)/2
            down = True
            total = 0

            for _ in range(len(self.weightList)):
                self.probs.append(value)
                total += value

                if down:
                    value -= 1
                    if value == middle:
                        down = False
                else:
                    value += 1

        self.probs = map(lambda a: float(a)/float(total),self.probs)
        
    def updateWeights(self, weights = None):
        if weights is None:
            weights = self._getWeightsPair()
        Settings.config.set("mogp_"+self.domainString,"weights","{0:.2f} {1:.2f}".format(weights[0],weights[1]))
        logger.info("MOGP weights set to {}".format(weights))
        
    
    def _getWeightsPair(self):
        w = [0,0]

        w[0] = Settings.random.choice(self.weightList, p=self.probs)
        w[1] = 1-w[0]
        
        return w 