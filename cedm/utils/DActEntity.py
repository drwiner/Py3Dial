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
DiaAct.py - dialogue act specification that extends dact.py
===========================================================

Copyright CUED Dialogue Systems Group 2015, 2016

**Basic Usage**: 
    >>> import DiaAct   
   
.. seealso:: CUED Imports/Dependencies: 

    import :class:`utils.ContextLogger` |.|
    import :class:`dact`

************************

'''

from utils.DiaAct import DiaActWithProb, DiaAct
from utils import dact
from utils import ContextLogger
import copy, re
logger = ContextLogger.getLogger('')

entitySeparator = '#'

class DiaActEntity(DiaActWithProb):
    
    def __init__(self, act = None, _entityName = None):
        
        
        
        if act is None:
            self.act = None
            self.items = []
            self.P_Au_O = 0.0
            self.entityname = None
        elif isinstance(act, DiaActEntity):
            self.act = copy.deepcopy(act.act)
            self.items = copy.deepcopy(act.items)
            self.P_Au_O = act.P_Au_O
            if act.entityname is not None:
                self.entityname = copy.deepcopy(act.entityname)
            else:
                self.entityname = _entityName
        else:
            if isinstance(act,DiaAct):
                if isinstance(act,DiaActWithProb):
                    self.act = copy.deepcopy(act.act)
                    self.items = copy.deepcopy(act.items)
                    self.P_Au_O = act.P_Au_O
                else:
                    self.act = copy.deepcopy(act.act)
                    self.items = copy.deepcopy(act.items)
                    self.P_Au_O = 1.0
            else:
                dia_act = self._InParseAct(act)
                self.act = dia_act['act']
                self.items = dia_act['slots']
                self.P_Au_O = 1.0
            
            self.entityname = _entityName
            self.items = map(lambda x: DactItemEntity.transform_item(x, _entityName),self.items)    
            
    def addItem(self, item):
        self.items.append(copy.deepcopy(item))
            
    def append(self, slot, value, negate=False):
        '''
        Add item to this act avoiding duplication

        :param slot: None
        :type slot: str
        :param value: None
        :type value: str
        :param negate: semantic operation is negation or not
        :type negate: bool [Default=False]
        :return:
        '''
        op = '='
        if negate:
            op = '!='
        self.items.append(DactItemEntity(slot, op, value, self.entityname))

    def append_front(self, slot, value, negate=False):
        '''
        Add item to this act avoiding duplication

        :param slot: None
        :type slot: str
        :param value: None
        :type value: str
        :param negate: operation is '=' or not?  False by default.
        :type negate: bool
        :return:
        '''
        op = '='
        if negate:
            op = '!='
        self.items = [DactItemEntity(slot, op, value)] + self.items

    def get_value(self, slot, negate=False):
        '''
        :param slot: slot name
        :type slot: str
        :param negate: relating to semantic operation, i.e slot = or slot !=.
        :type negate: bool - default False
        :returns: (str) value
        '''
        value = None
        for item in self.items:
            if item.is_relation():
                continue # skip relations
            if slot == item.slot and negate == (item.op == '!='):
                if value is not None:
                    logger.warning('DiaAct contains multiple values for one slot: ' + str(self))
                else:
                    value = item.val
        return value
    
    def find_relation(self, slot, negate=False):
        '''
        :param slot: slot name
        :type slot: str
        :param negate: relating to semantic operation, i.e slot = or slot !=.
        :type negate: bool - default False
        :returns: (str) value
        '''
        rel = None
        for item in self.items:
            if item.is_relation() and (item.slot == slot or item.val == slot):
                if rel is not None:
                    logger.warning('DiaAct contains multiple relations for one slot: ' + str(self))
                else:
                    rel = item
        return rel
            
    def contains_relation(self):
        for item in self.items:
            if item.is_relation():
                return True
                
        return False
            
    def to_string(self):
        '''
        :param None:
        :returns: (str) semantic act
        '''
        s = ''
        s += self.act + '('
        for i, item in enumerate(self.items):
            if i != 0:
                s += ','

            if item.slot is not None:
                if not isinstance(item, DactItemEntity) or item.slot_entity is None:
                    s += '?' + entitySeparator
                else:
                    s += str(item.slot_entity) + entitySeparator
                s += item.slot
            if item.val is not None:
                s += item.op
                if isinstance(item, DactItemEntity) and item.value_entity is not None:
                    s += str(item.value_entity) + entitySeparator + str(item.val)
                else:
                    s += '"'+str(item.val)+'"'
        s += ')'
        return s
    
    def to_string_plain(self):
        return super(DiaActEntity, self).to_string()
    
    def separate(self, lastSysAct = None):
        # how to handle affirm(..) and negate(..)
        
        newAct = self.act
        
        eact = None
        ract = None
        if lastSysAct is not None:
            if isinstance(lastSysAct, DiaActEntity) and lastSysAct.contains_relation():
                if ('affirm' in self.act or 'negate' in self.act) and ract is None:
                    ract = DiaActEntity()
                    ract.act = self.act
                    ract.P_Au_O = self.P_Au_O
                    ract.entityname = "{}#{}".format(lastSysAct.items[0].slot_entity,lastSysAct.items[0].value_entity)
                    newAct = 'inform'
                elif lastSysAct.act == 'confirm' and self.act == 'inform' and ract is None:
                    for lItem in lastSysAct.items:
                        if lItem.is_relation() and self.get_value(lItem.slot) is not None:
                            ract = DiaActEntity()
                            ract.act = 'negate'
                            ract.P_Au_O = self.P_Au_O
                            ract.entityname = "{}#{}".format(lastSysAct.items[0].slot_entity,lastSysAct.items[0].value_entity)
            else:
                if  ('affirm' in self.act or 'negate' in self.act) and eact is None:
                    eact = DiaActEntity()
                    eact.act = self.act
                    eact.P_Au_O = self.P_Au_O
                    eact.entityname = self.entityname
                    newAct = 'inform'
                elif lastSysAct.act == 'confirm' and self.act == 'inform' and eact is None:
                    for lItem in lastSysAct.items:
                        if not lItem.is_relation() and self.find_relation(lItem.slot) is not None:
                            eact = DiaActEntity()
                            eact.act = 'negate'
                            eact.P_Au_O = self.P_Au_O
                            eact.entityname = self.entityname
        
        if len(self.items):
            for item in self.items:
                if isinstance(item, DactItemEntity) and item.is_relation():
                    # defines a relation to another entityname
                    if ract is None:
                        ract = DiaActEntity()
                        ract.act = newAct
                        ract.P_Au_O = self.P_Au_O
                        ract.entityname = "{}#{}".format(item.slot_entity,item.value_entity)
                    ract.items.append(copy.deepcopy(item))
                else:
                    # defines a value
                    if eact is None:
                        eact = DiaActEntity()
                        eact.act = newAct
                        eact.P_Au_O = self.P_Au_O
                        eact.entityname = self.entityname
                    eact.items.append(copy.deepcopy(item))
        elif eact is None:
            eact = DiaActEntity()
            eact.act = newAct
            eact.P_Au_O = self.P_Au_O
            eact.entityname = self.entityname
        
        return eact, ract
    
    def _InParseAct(self, t):
        r = {}
        r['slots'] = []
    
        if t == "BAD ACT!!":
            r['act'] = 'null'
            return r
    
        #m = re.search('^(.*)\((.*)\)$',t.strip())
        m = re.search('^([^\(\)]*)\((.*)\)$',t.strip())
        if not m:
            r['act'] = 'null'
            return r
    
        r['act'] = m.group(1).strip()
        content = m.group(2)
        while len(content) > 0:
            m = re.search('^([^,!=]*)(!?=)\s*\"([^\"]*)\"\s*,?', content)
            if m:
                slot = m.group(1).strip()
                op = m.group(2).strip()
                val = m.group(3).strip("' ")
                items = DactItemEntity(slot, op, val)
                content = re.sub('^([^,!=]*)(!?=)\s*\"([^\"]*)\"\s*,?', '', content)
                r['slots'].append(items)
                continue
            m = re.search('^([^,=]*)(!?=)\s*([^,]*)\s*,?', content)
            if m:
                slot = m.group(1).strip()
                op = m.group(2).strip()
                val = m.group(3).strip("' ")
                items = DactItemEntity(slot, op, val)
                content = re.sub('^([^,=]*)(!?=)\s*([^,]*)\s*,?', '', content)
                r['slots'].append(items)
                continue
            m = re.search('^([^,]*),?', content)
            if m:
                slot = m.group(1).strip()
                op = None
                val = None
                items = DactItemEntity(slot, op, val)
                content = re.sub('^([^,]*),?', '', content)
                r['slots'].append(items)
                continue
            raise RuntimeError, 'Cant parse content fragment: %s' % content
    
        return r
            
            
class DactItemEntity(dact.DactItem):
    
    def __init__(self, slot, op, val, slot_e = None, value_e = None):
        self.slot, self.slot_entity = DactItemEntity._split(slot, slot_e)
        self.op = op
        self.val, self.value_entity = DactItemEntity._split(val, value_e)
        
        
    def is_relation(self):
        return self.value_entity is not None and self.val is not None
    
    def get_other_entity(self, e):
        if self.slot_entity == e:
            return self.value_entity
        elif self.value_entity == e:
            return self.slot_entity
        else:
            return None
        
    @staticmethod    
    def transform_item(item, slot_e = None, value_e = None):
        if isinstance(item,DactItemEntity):
            if item.slot_entity is not None:
                slot_e = item.slot_entity
            if item.value_entity is not None:
                value_e = item.value_entity
        slot, slot_e = DactItemEntity._split(item.slot, slot_e)
        op = item.op
        val, value_e = DactItemEntity._split(item.val, value_e)
        
        
#         slot = item.slot
#         op = item.op
#         val = item.val
#         
#         if slot is not None:
#             slot_split = slot.split(entitySeparator)
#             if len(slot_split) > 1:
#                 slot = slot_split[1]
#                 new_slot_e = slot_split[0]
#                 if slot_e is not None and new_slot_e != slot_e:
#                     logger.error("Mismatch between entities of slot of dialogue act.")
#                 slot_e = new_slot_e
#         
#         if val is not None:
#             val_split = val.split(entitySeparator)
#             if len(val_split) > 1:
#                 val = val_split[1]
#                 new_val_e = val_split[0]
#                 if value_e is not None and new_val_e != value_e:
#                     logger.error("Mismatch between entities of value of dialogue act.")
#                 value_e = new_val_e
        
        return DactItemEntity(slot, op, val, slot_e, value_e)
    
    @staticmethod
    def _split(s, c = None):
        if s is None:
            return None, c
        if isinstance(s,bool):
            print "What"
        s_split = s.split(entitySeparator)
        if len(s_split) > 1:
            s = s_split[1]
            s_e = s_split[0]
            if c is not None and s_e != c:
                logger.error("Mismatch between entities of value of dialogue act.")
            return s, s_e
        else:
            return s, c
        
    def match(self, other):
        '''
        Commutative operation for comparing two items.
        Note that "self" is the goal constraint, and "other" is from the system action.
        The item in "other" must be more specific. For example, the system action confirm(food=dontcare) doesn't match
        the goal with food=chinese, but confirm(food=chinese) matches the goal food=dontcare.

        If slots are different, return True.

        If slots are the same, (possible values are x, y, dontcare, !x, !y, !dontcare)s
            x, x = True
            x, y = False
            dontcare, x = True
            x, dontcare = False
            dontcare, dontcare = True

            x, !x = False
            x, !y = True
            x, !dontcare = True
            dontcare, !x = False
            dontcare, !dontcare = False

            !x, !x = True
            !x, !y = True
            !x, !dontcare = True
            !dontcare, !dontcare = True

        :param other:
        :return:
        '''
        if not self.is_relation() and not isinstance(other, DactItemEntity):
            return super(DactItemEntity, self).match(other)
        
        if (not isinstance(other, DactItemEntity) or not other.is_relation()) and not self.is_relation():
            return super(DactItemEntity, self).match(other)
        
        if not self.is_relation() or (not isinstance(other, DactItemEntity) or not other.is_relation()):
            # if at least one is not a relation (and the other one is, otherwise super had been called above) they match
            return True
        
        # only if entities and slots match, op matters. otherwise this matches
        # hence we have to consider two cases:
        if self.slot_entity == other.slot_entity and self.slot == other.slot and self.val == other.val and self.value_entity and other.value_entity:
            # both are aligned
            return self.op == other.op
        elif self.slot_entity == other.value_entity and self.slot == other.val and self.val == other.slot and self.value_entity and other.slot_entity:
            # both are cross-aligned
            return self.op == other.op
        else:
            # in all other cases they match
            return True        

    def __eq__(self, other):
        if not isinstance(other, DactItemEntity):
            return False
        return self.slot == other.slot and self.op == other.op and self.val == other.val and self.slot_entity == other.slot_entity and self.value_entity == other.value_entity
        
if __name__ == "__main__":
    act = "inform(e1" + entitySeparator + "area = E2" + entitySeparator + "area, e1" + entitySeparator + "pricerange = cheap)"
    a = DiaActEntity(act, "e1")
    print a