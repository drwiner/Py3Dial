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

import sys
import numpy as np

import ontology.FlatOntologyManager as FlatOnt


def get_feudal_masks(non_exec, slots, slot_independent_actions, slot_specific_actions):

    feudal_masks = {'req_info': {}, 'give_info': None, 'master': None}
    give_info_masks = np.zeros(len(slot_independent_actions))
    give_info_masks[-1] = -sys.maxint
    for i, action in enumerate(slot_independent_actions):
        if action in non_exec:
            give_info_masks[i] = -sys.maxint
    feudal_masks['give_info'] = give_info_masks
    for slot in slots:
        feudal_masks['req_info'][slot] = np.zeros(len(slot_specific_actions))
        feudal_masks['req_info'][slot][-1] = -sys.maxint
        for i, action in enumerate(slot_specific_actions):
            if action == 'reqmore':
                if action in non_exec:
                    feudal_masks['req_info'][slot][i] = -sys.maxint
            elif action + '_' + slot in non_exec:
                feudal_masks['req_info'][slot][i] = -sys.maxint
    master_masks = np.zeros(3)
    master_masks[:] = -sys.maxint
    if 0 in give_info_masks:
        master_masks[0] = 0
    for slot in slots:
        if 0 in feudal_masks['req_info'][slot]:
            master_masks[1] = 0
    feudal_masks['master'] = master_masks
    # print(non_exec)
    # print(feudal_masks)
    return feudal_masks

def get_feudalAC_masks(non_exec, slots, slot_independent_actions, slot_specific_actions):

    feudal_masks = {'req_info': {}, 'give_info': None, 'master': None}
    give_info_masks = np.zeros(len(slot_independent_actions))
    give_info_masks[-1] = -sys.maxint
    for i, action in enumerate(slot_independent_actions):
        if action in non_exec:
            give_info_masks[i] = -sys.maxint
    feudal_masks['give_info'] = give_info_masks
    for slot in slots:
        feudal_masks['req_info'][slot] = np.zeros(len(slot_specific_actions))
        feudal_masks['req_info'][slot][-1] = -sys.maxint
        for i, action in enumerate(slot_specific_actions):
            if action + '_' + slot in non_exec:
                feudal_masks['req_info'][slot][i] = -sys.maxint
    master_masks = np.zeros(len(slots)+2)
    master_masks[:] = -sys.maxint
    if 0 in give_info_masks:
        master_masks[-2] = 0
    for i, slot in enumerate(slots):
        if 0 in feudal_masks['req_info'][slot]:
            master_masks[i] = 0
    feudal_masks['master'] = master_masks
    # print(non_exec)
    # print(feudal_masks)
    return feudal_masks


def get_feudal_slot_mask(non_exec, slot, slot_actions):
    slot_masks = np.zeros(len(slot_actions))
    slot_masks[-1] = -sys.maxint
    if slot == 'master' or slot == 'give_info':
        for i, action in enumerate(slot_actions):
            if action in non_exec:
                slot_masks[i] = -sys.maxint
    else:
        for i, action in enumerate(slot_actions):
            action = action+'_'+slot
            if action in non_exec:
                slot_masks[i] = -sys.maxint
    return slot_masks