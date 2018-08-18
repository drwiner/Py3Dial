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
SemI.py - Semantic input parser
===================================

Copyright CUED Dialogue Systems Group 2015, 2016

.. seealso:: CUED Imports/Dependencies:

    import :mod:`ontology.OntologyUtils` |.|
    import :mod:`utils.DiaAct` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger`

************************

'''


__author__ = "cued_dialogue_systems_group"


from utils import ContextLogger, Settings
from ontology import OntologyUtils
import SemIContextUtils as contextUtils
from cedm.utils.DActEntity import DiaActEntity
logger = ContextLogger.getLogger('')

class SemIManager(object):
    '''
    The Semantic Input Manager contains a dictionary with all the SemI objects currently running, the key of the dictionary is the
    domain tag

    '''

    def __init__(self):
        '''
        When Initialised it tries to load all the domains, in the case there are configuration problems these SemI objects are not loaded.
        :return:
        '''
        self.SPECIAL_TYPES = ['topicmanager','wikipedia','ood']
        self.semiManagers = dict.fromkeys(OntologyUtils.available_domains)
        for domainTag in self.semiManagers.keys():
            try:
                self.semiManagers[domainTag] = self._load_domains_semi(dstring=domainTag)
            except AttributeError:
                continue

        

    def _ensure_booted(self, domainTag):
        '''
        Boots up the semi ability for the specified domain if required
        
        :param domainTag: domain description
        :type domainTag: str
        :return: None
        '''
        if self.semiManagers[domainTag] is None:
            self.semiManagers[domainTag] = self._load_domains_semi(dstring=domainTag)
        return

    def _load_domains_semi(self, dstring):
        '''
        Get from the config file the SemI choice of method for this domain and load it.
        If you want to add a new semantic input parser you must add a line in this method calling explicitly
        the new SemI class, that must inherit from SemI
        
        .. Note:
            To dynamically load a class, the __init__() must take one argument: domainString.

        :param dstring: the name of the domain
        :type dstring: str
        :return: the class with the SemI implementation
        '''


        # 1. get type:
        semi_type = 'PassthroughSemI'  # domain+resource independent default
        if Settings.config.has_option('semi_'+dstring, 'semitype'):
            semi_type = Settings.config.get('semi_'+dstring, 'semitype')
            
        parsing_method = None
        
        if dstring in self.SPECIAL_TYPES:
            semi_type = 'RegexSemI'
        # 2. And load that method for the domain:
        if semi_type == 'PassthroughSemI':
            from semi.RuleSemIMethods import PassthroughSemI
            parsing_method = PassthroughSemI()
        elif semi_type == 'RegexSemI':
            from semi.RuleSemIMethods import RegexSemI
            parsing_method = RegexSemI(domainTag=dstring)
        else:
            try:
                # try to view the config string as a complete module path to the class to be instantiated
                components = semi_type.split('.')
                packageString = '.'.join(components[:-1]) 
                classString = components[-1]
                mod = __import__(packageString, fromlist=[classString])
                klass = getattr(mod, classString)
                parsing_method = klass(dstring)
            except ImportError:
                logger.warning('Unknown semantic decoder "{}" for domain "{}". Using PassthroughSemI.'.format(semi_type, dstring))
                from RuleSemIMethods import PassthroughSemI
                parsing_method = PassthroughSemI()

        return parsing_method
            
    # Added turn as an optional argument, only needed by DLSemI
    def decode(self, ASR_obs, sys_act, domainTag, turn=None):
        '''
        The main method for semantic decoding. It takes the ASR input and returns a list of semantic interpretations. To decode, the task is delegated to the respective domain semantic decoder
        
        :param ASR_obs: ASR hypotheses
        :type ASR_obs: list
        :param sys_act: is the system action prior to collecting users response in obs.
        :type sys_act: str
        :param domainTag: is the domain we want to parse the obs in
        :type domainTag: str
        :param turn: the turn id, this parameter is optional
        :type turn: int
        :return: None
        '''

        # use the correct domain:
        self.active_domain = domainTag # used down call chain in adding context
        self._ensure_booted(domainTag)
        
        # --------
        # explore how to clean IBM asr - "i dont care" problem ...    
        for i in range(len(ASR_obs)):            
            was = ASR_obs[i][0]
            fix = [str(c) for c in was if c.isalpha() or c.isspace() or c=="'" or c!="!" or c!="?"]     # slow way - initial fix #lmr46 06/09/16 this filtering filters question marks that are important!!!! syj requirement
            res = ''.join(fix)
            ASR_obs[i] = (res.rstrip(), ASR_obs[i][1])
        #---------------------------------------------------  
        
        # Additional argument turn as described above
        hyps = self.semiManagers[domainTag].decode(ASR_obs, sys_act, turn=turn)
        logger.info(hyps)
        # add context if required
        hyps = contextUtils._add_context_to_user_act(sys_act,hyps,self.active_domain)
        return hyps

    def clean_possible_texthub_switch(self,userActText):
        '''
        NB: only for texthub.py
        
        This removes switch("Domain") - as you may enter in texthub if using the switch topic tracker
        You can add domain information after e.g.: switch("CamRestaurants")i want a cheap restaurant
        
        :param userActText: list of user act hypothesis?
        :return:
        '''

        text_first_hyp = userActText[0][0]    # userActText is [('switch("CamRestaurants")',1.0)]
        if 'switch("' in text_first_hyp:
            tmp = text_first_hyp.split('"') 
            if len(tmp) > 2 and len(tmp[2]): # remove the ) in switch("CamRestaurants")
                assert(tmp[2][0]==')') 
                tmp[2] = tmp[2][1:]   
            cleaned = "".join(tmp[2:])            
            return [(cleaned,1.0)]      # TODO -- will need fixing if simulated errors are introduced into texthub
        return userActText
    
    
    def simulate_add_context_to_user_act(self, sys_act, user_acts, eType):
        '''
        NB: only for Simulate.py
        
        While for simulation no semantic decoding is needed, context information needs to be added in some instances. This is done with this method.
        
        :param sys_act: the last system act
        :type sys_act: str
        :param user_acts: user act hypotheses
        :type user_acts: list
        :param domainTag: the domain of the dialogue
        :type domainTag: str
        '''
        # simulate - will only pass user_act, and we will return contextual user act
        
        hyps = contextUtils._add_context_to_user_act(sys_act.to_string_plain(),hyps=user_acts,active_domain=eType)
        return hyps  # just the act
    
    def simulate_add_context_to_user_act_relation(self, sys_act, user_acts, rType):
        '''
        NB: only for Simulate.py
        
        While for simulation no semantic decoding is needed, context information needs to be added in some instances. This is done with this method.
        
        :param sys_act: the last system act
        :type sys_act: str
        :param user_acts: user act hypotheses
        :type user_acts: list
        :param domainTag: the domain of the dialogue
        :type domainTag: str
        '''
        # simulate - will only pass user_act, and we will return contextual user act
        
        logger.info('Possibly adding context to user semi hyps: %s' % user_acts)
        if not len(user_acts) or sys_act is None:
            return user_acts
        # if negated -- only deal with this for now if it pertains to a binary slot. dont have an act for
        # "i dont want indian food" for example
        new_hyps = []
        for hyp in user_acts:
            if hyp.act in ['affirm()','negate()']:
                user_act = hyp.act
                user_act = self._convert_yes_no_relation(sys_act, user_act)  
                user_act.P_Au_O = hyp.P_Au_O
                new_hyps.append(user_act)
            else:
                new_hyps.append(hyp)
        return new_hyps
    
    def _convert_yes_no_relation(self, sys_act, user_act):
        '''
        Converts yes/no only responses from user into affirm and negate.
        
        Necessary for binary slots in system utterance ie. request(hasparking) --> inform(slot=opposite)
        
        :param sys_act: the last system action
        :type sys_act: str
        :param user_act: the user input act to be processed
        :type user_act: str
        :return: the transformed user act if conditions apply else the untouched user act
        '''
    
        # TODO - should definitely be more scenarios to deal with here
        dact = DiaActEntity(sys_act)
    #     dact = sys_act
        slot_val_pairs = []
        if dact.act in ['request', 'confirm']: 
            for item in dact.items:         
                slot, op, val = item.slot, item.op, item.val
                if dact.act == 'request':# and slot not in OntologyUtils.BINARY_SLOTS[active_domain]:
                    logger.warning('Attempting to negate/affirm a non binary valued slot')  
                    return user_act # default back to unchanged act
                if dact.act == 'confirm' and user_act.act == 'negate()':
                    # cases 1: should have been parsed by semi into some deny(A=b,A=c) act
                    # cases 2: user just replied 'no' which doesn't make sense -- just return original act
                    # Dangerously close to writing policy rules here ... dont want to assume we can return 
                    # inform(A=dontcare) for instance. Same for above if statement. Let the dialog have a bad turn... 
                    return user_act  
                val = self._apply_affirm_negate_to_value_relation(user_act, val)
                slot_val_pairs.append(self._give_context(slot,val))
            contextual_act = DiaActEntity('inform('+','.join(slot_val_pairs)+')')
            logger.info("New contexual act: "+contextual_act)
            return unicode(contextual_act)
        elif dact.act in ['reqmore']:
            if user_act.act == 'negate()':
                return user_act  # TODO - think it might cause problems in MTurk dialogs to return bye()
                #return unicode('bye()')  # no to reqmore() is an implicit goodbye. 
        else:
            logger.warning('affirm or negate in response to currently unhandled system_act:\n '+str(sys_act))
        return user_act  # otherwise leave it the same
    
    def _apply_affirm_negate_to_value_relation(self, affirm_or_negate,val):
        '''
        Returns the value implied by the affirm() or negate() act
        
        Returns the passed value val as a default
        
        :param affirm_or_negate: the affirm() or negate() act
        :type affirm_or_negate: 
        :param val: the value to be used
        :type val: str
        :return: value or 0/1
        '''
    
    #         NB we have checked at this point that the slotX is binary if the sys_act was request(slotX).
    #         For confirm(slotX=Y) we can only deal with this if it is an affirm() --> inform(slotX=Y)
    #         if it is a negate we will need to have checked this outside this function.
    
    
        if val is None:
            val = '1'     # binary slot true value - should be only case where val is None here 
        if affirm_or_negate.act == 'affirm()':
            return val
        elif affirm_or_negate.act == 'negate()' and val in ['0','1']: #only know how to negate binary slots
            # Note that request(hasparking) > negate() > val would have been None on passing here (changed to 1 above)
            return '0' if val == '1' else '1'
        else: 
            return val  # set val = none ? so we ask again on this slot?
    
    def _give_context(self, slot,val):
        '''
        Added context to user act, ie, create the slot-value string.
        
        :param slot: slot
        :type slot: str
        :param val: value
        :type val: str
        :return: the string slot='value'
        '''
        
        contextual_slot_val_pair = slot+'="'+val+'"'
        logger.info("Added context to user act: "+contextual_slot_val_pair) 
        return contextual_slot_val_pair
    
#END OF FILE
