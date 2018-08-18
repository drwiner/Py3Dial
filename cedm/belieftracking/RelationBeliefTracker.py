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

from baseline import RuleBasedTracker, Uacts, labels, normalise_dict, clip
import copy, math, pprint
from collections import defaultdict
from ontology import Ontology
import belieftracking.BeliefTrackingUtils as BeliefTrackingUtils
from cedm.utils import dact
from utils import ContextLogger, Settings

logger = ContextLogger.getLogger('') 



class RelationFocusTracker(RuleBasedTracker):
    """
    It accumulates evidence and has a simple model of how the state changes throughout the dialogue.
    Only track goals but not requested slots and method.
    """
    def __init__(self,eType):
        super(RelationFocusTracker, self).__init__(eType)
        self.restart()

    def _addTurn(self, turn, relation):
        '''
        Add turn info

        :param turn:
        :type turn: dict
        
        :return: None
        '''
        hyps = copy.deepcopy(relation.lastHyps)
        if "dialog-acts" in turn["output"] :
            mact = turn["output"]["dialog-acts"]
        else :
            mact = []
        slu_hyps = Uacts(turn)
       
        this_u = defaultdict(lambda : defaultdict(float))
        this_deny = defaultdict(lambda : defaultdict(float))
        method_stats = defaultdict(float)
        requested_slot_stats = defaultdict(float)
        discourseAct_stats = defaultdict(float)
        for score, uact in slu_hyps :
            informed_goals, denied_goals, requested, method, discourseAct, _ = labels(uact, mact, None)
            method_stats[method] += score
            for slot in requested:
                requested_slot_stats[slot] += score
            # goal_labels
            for slot in informed_goals:
                this_u[slot][informed_goals[slot]] += score
            for slot in denied_goals:
                for i in range(0,len(denied_goals[slot])):
                    this_deny[slot][denied_goals[slot][i]] += score
                    this_u[slot][denied_goals[slot][i]] += 0.0 # ensure there is an entry
            discourseAct_stats[discourseAct] += score


        for slot in set(this_u.keys() + hyps["goal-labels"].keys()) :
            q = max(0.0,1.0-sum([this_u[slot][value] for value in this_u[slot]])) # clipping at zero because rounding errors
            q = max(0.0,q-sum([this_deny[slot][value] for value in this_deny[slot]])) # also use probability mass in denied
            
            if slot not in hyps["goal-labels"] :
                hyps["goal-labels"][slot] = {}
                
            for value in hyps["goal-labels"][slot] :
                
                hyps["goal-labels"][slot][value] *= q
            prev_values = hyps["goal-labels"][slot].keys()
            for value in this_u[slot] :
                if value in prev_values :
                    hyps["goal-labels"][slot][value] += this_u[slot][value]
                else :
                    hyps["goal-labels"][slot][value]=this_u[slot][value]
        
            hyps["goal-labels"][slot] = normalise_dict(hyps["goal-labels"][slot])
        
        # method node, in 'focus' manner:
        q = min(1.0,max(0.0,method_stats["none"]))
        method_label = hyps["method-label"]
        for method in method_label:
            if method != "none" :
                method_label[method] *= q
        for method in method_stats:
            if method == "none" :
                continue
            if method not in method_label :
                method_label[method] = 0.0
            method_label[method] += method_stats[method]
        
#         if "none" not in method_label :
        method_label["none"] = max(0.0, 1.0-sum(method_label.values()))
        
        hyps["method-label"] = normalise_dict(method_label)

        # discourseAct (is same to non-focus)
        hyps["discourseAct-labels"] = normalise_dict(discourseAct_stats)

        # requested slots
        informed_slots = []
        for act in mact :
            if act["act"] == "inform" :
                for slot,value in act["slots"]:
                    informed_slots.append(slot)
                    
        for slot in set(requested_slot_stats.keys() + hyps["requested-slots"].keys()):
            p = requested_slot_stats[slot]
            prev_p = 0.0
            if slot in hyps["requested-slots"] :
                prev_p = hyps["requested-slots"][slot]
            x = 1.0-float(slot in informed_slots)
            new_p = x*prev_p + p
            hyps["requested-slots"][slot] = clip(new_p)
            
        relation.lastHyps = hyps 
        return hyps
    
    
#     def restart(self):
#         '''
#         Reset the hypotheses
#         '''
#         super(FocusTracker, self).restart()
#         self.hyps = {"goal-labels":{},"method-label":{}, "requested-slots":{}}

    def _tobelief(self, relation, track):
        '''
        Add up previous belief and current tracking result to current belief

        :param prev_belief:
        :type prev_belief: dict

        :param track:
        :type track: dict
       
        :return: dict -- belief state
        '''
        prev_belief = relation.belief
        belief = {}
        rel_slots = Ontology.global_ontology.get_common_slots(relation.e1,relation.e2)
        belief['requested'] = {}
        for slot in rel_slots:
            if slot in track['goal-labels']:
                infom_slot_vals = ['=']
                # su259: user simulator may issue a dontcare for all informable slots, not only system requestable
#                 if slot not in Ontology.global_ontology.get_system_requestable_slots(self.eType):
#                     belief[slot] = dict.fromkeys(infom_slot_vals, 0.0)
#                 else:
                
                # keep dontcare for relations for now
                belief[slot] = dict.fromkeys(infom_slot_vals+['dontcare'], 0.0)
                for v in track['goal-labels'][slot]:
                    belief[slot][v] = track['goal-labels'][slot][v]
                belief[slot]['**NONE**'] = 1.0 - sum(belief[slot].values())
            else:
                belief[slot] = prev_belief['beliefs'][slot]
            belief['requested'].update({slot:0.0})
                
#         belief['method'] = dict.fromkeys(Ontology.global_ontology.get_method(self.eType), 0.0)
#         for v in track['method-label']:
#             belief['method'][v] = track['method-label'][v]
#         belief['discourseAct'] = dict.fromkeys(Ontology.global_ontology.get_discourseAct(self.eType), 0.0)
#         for v in track['discourseAct-labels']:
#             belief['discourseAct'][v] = track['discourseAct-labels'][v]
#         belief['requested'] = {'=':0.0}
        for v in track['requested-slots']:
            belief['requested'][v] = track['requested-slots'][v]

        return {'beliefs': belief}
    
    def _convertHypToTurn(self, lastact, obs):
        '''
        Convert hypotheses to turn
        
        :param lastact: last system dialgue act
        :type lastact: string

        :param obs: current observation
        :type obs: list
        
        :return: dict -- turn dict
        '''
        curturn = {'turn-index': self.turn}

        # Last system action
        slastact = []
        if self.turn > 0:
            if lastact.contains_relation():
                slastact = dact.ParseActRel(lastact, user=False)
            else:
                slastact = dact.ParseAct(lastact, user=False)
            slastact = BeliefTrackingUtils._transformAct(slastact, {}, 
                                                         Ontology.global_ontology.get_ontology(self.eType), 
                                                         user=False)
        curturn['output'] = {'dialog-acts': slastact}

        # User act hyps
        accumulated = defaultdict(float)
        for ob in obs:
            prob = ob.P_Au_O
            hyp = dact.ParseActRel(ob)
            hyp = BeliefTrackingUtils._transformAct(hyp, {}, Ontology.global_ontology.get_ontology(self.eType))
            hyp = dact.inferSlotsForAct(hyp)

            prob = min(1.0, prob)
            if prob < 0:
                prob = math.exp(prob)
            accumulated = BeliefTrackingUtils._addprob(accumulated, hyp, prob)
        sluhyps = BeliefTrackingUtils._normaliseandsort(accumulated)

        curturn['input'] = {'live': {'asr-hyps':[], 'slu-hyps':sluhyps}}
        return curturn
    
    def update_belief_state(self, relation, lastact):
        '''
        Does the actual belief tracking via tracker.addTurn

        :param lastact: last system dialgoue act
        :type lastact: string

        :param obs: current observation
        :type obs: list

        :return: dict -- previous belief state
        '''
        userActs = relation.getUserInput()
        curturn = self._convertHypToTurn(lastact, userActs)
        if self.turn == 0:
            relation.belief = self._init_belief(relation)

        relation.belief = self._updateBelief(curturn, relation)
        relation.updateUserInput(userActs) # hack as whole dict is overwritten by previous line
#         self._print_belief()

        logger.debug(pprint.pformat(curturn))

        self.turn += 1
        logger.debug(self.str(relation))
        
#         self._printTopBeliefs(relation)
        
        return
    
    def _init_belief(self, relation):
        '''
        Simply constructs the belief state data structure at turn 0

        :param constraints: a dict of constraints
        :type constraints: dict
       
        :return: dict -- initiliased belief state 
        ''' 
        rel_slots = Ontology.global_ontology.get_common_slots(relation.e1,relation.e2)
        belief = {} 
        belief['requested'] = {}
        for slot in rel_slots:
            inform_slot_vals = ['=']
            belief[slot] = dict.fromkeys(inform_slot_vals+['dontcare'], 0.0)
            belief[slot]['**NONE**'] = 1.0
            belief['requested'].update({slot : 0.0})  
#         belief['method'] = dict.fromkeys(Ontology.global_ontology.get_method(self.eType), 0.0)
#         belief['method']['none'] = 1.0
#         belief['discourseAct'] = dict.fromkeys(Ontology.global_ontology.get_discourseAct(self.eType), 0.0)
#         belief['discourseAct']['none'] = 1.0
#         belief['requested'] = {'=' : 0.0}
#         if constraints is not None:
#             belief = self._conditionally_init_belief(belief,constraints)
        return {'beliefs': belief}
    
#     def _conditionally_init_belief(self,belief,constraints):
#         """
#         Method for conditionally setting up the inital belief state of a domain based on information/events that occured
#         earlier in the dialogue in ANOTHER (ie different) domain.
#        
#         :param belief: initial belief state
#         :type belief: dict
# 
#         :param constraints: a dict of constraints
#         :type constraints: dict
#        
#         :return: None        
#         """ 
#         # Now initialise the BELIEFS in this domain, based on the determine prior domain constraints
#         for slot,valList in constraints.iteritems(): 
#             if valList is not None and slot not in ['name']:
#                 prob_per_val = self.CONDITIONAL_BELIEF_PROB/float(len(set(valList))) 
#                 for val in valList:
#                     belief[slot][val] = prob_per_val
#                 # and now normalise (plus deal with **NONE**)
#                 num_zeros = belief[slot].values().count(0.0)  #dont need a -1 for the **NONE** value as not 0 yet
#                 prob_per_other_val = (1.0-self.CONDITIONAL_BELIEF_PROB)/float(num_zeros)
#                 for k,v in belief[slot].iteritems():
#                     if v == 0.0:
#                         belief[slot][k] = prob_per_other_val  #cant think of better way than to loop for this...
#                 belief[slot]['**NONE**'] = 0.0
#         #TODO - delete debug prints: print belief
#         #print constraints
#         #raw_input("just cond init blief")
#         return belief
    
    
            
            
if __name__ == '__main__':
    from utils.er_model.Belief import Relation, Entity
    from utils.er_model.DActEntity import DiaActEntity
    
    
    Settings.load_root('/Users/su259/cued/pydial/')
    Settings.load_config(None)
    Settings.config.add_section("GENERAL")
    Settings.config.set("GENERAL",'domains', 'CamRestaurants,CamHotels')
    
    Ontology.init_global_ontology()
    
    e1 = Entity('CamRestaurants')
    e2 = Entity('CamHotels')
    rel = Relation(e1,e2)
    rel.updateUserInput([(DiaActEntity('inform(CamRestaurants#area=CamHotels#area)'),1.0)])
#     rel.lastHyps = [(DiaActEntity('inform(CamRestaurants#area=CamHotels#area)'),1.0)]
#     rel.lastHyps = [(DiaActEntity('confirm(CamRestaurants#area=CamHotels#area)'),1.0)]
    tracker = RelationFocusTracker('relation')
    tracker.update_belief_state(rel, DiaActEntity('hello()'))
    rel.updateUserInput([(DiaActEntity('confirm(CamRestaurants#area=CamHotels#area)'),1.0)])
    tracker.update_belief_state(rel, DiaActEntity('hello()'))
    pass