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


import cedm.belieftracking.baseline as bs

from collections import defaultdict
import copy

discourseActs = ["ack", 
                "hello", 
                "none", 
                "repeat", 
                "silence", 
                "thankyou",
                "bye"]

class WorldFocusTracker(bs.RuleBasedTracker):
    """
    It accumulates evidence and has a simple model of how the state changes throughout the dialogue.
    Only track goals but not requested slots and method.
    """
    def __init__(self):
        super(WorldFocusTracker, self).__init__('TownInfo')
        self.restart()
        
    def _addTurn(self, turn, world):
        
        hyps = copy.deepcopy(world.lastHyps)
        if "dialog-acts" in turn["output"] :
            mact = turn["output"]["dialog-acts"]
        else :
            mact = []
        slu_hyps = bs.Uacts(turn)
        
        discourseAct_stats = defaultdict(float)
        
        for score, uact in slu_hyps :
            _, _, _, _, discourseAct, _ = bs.labels(uact, mact, None)
            # discourseAct
            discourseAct_stats[discourseAct] += score
            
        hyps["discourseAct-labels"] = bs.normalise_dict(discourseAct_stats)
        
        world.lastHyps = hyps 
        return hyps
    
    def _tobelief(self, world, track):
        '''
        Add up previous belief and current tracking result to current belief

        :param prev_belief:
        :type prev_belief: dict

        :param track:
        :type track: dict
       
        :return: dict -- belief state
        '''
        belief = {}
        belief['discourseAct'] = dict.fromkeys(discourseActs, 0.0)
        for v in track['discourseAct-labels']:
            belief['discourseAct'][v] = track['discourseAct-labels'][v]
        return {'beliefs': belief}
    
    def update_belief_state(self, world, lastact):
        '''
        Does the actual belief tracking via tracker.addTurn

        :param lastact: last system dialgoue act
        :type lastact: string

        :param obs: current observation
        :type obs: list
        
        :param constraints:
        :type constraints: dict

        :return: dict -- previous belief state
        '''
        curturn = self._convertHypToTurn(lastact, world.getUserInput())
        
        if self.turn == 0:
            world.belief = self._init_belief()

        world.belief = self._updateBelief(curturn, world)
#         self._print_belief()

        self.turn += 1
        
        return
    
    def _init_belief(self, constraints=None):
        '''
        Simply constructs the belief state data structure at turn 0

        :param constraints: a dict of constraints
        :type constraints: dict
       
        :return: dict -- initiliased belief state 
        ''' 
        belief = {} 
        belief['discourseAct'] = dict.fromkeys(discourseActs, 0.0)
        belief['discourseAct']['none'] = 1.0
        return {'beliefs': belief}
    
    
    
