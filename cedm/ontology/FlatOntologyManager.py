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
FlatOntologyManager.py - Domain class and Multidomain API
==========================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

Controls Access to the ontology files.

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger` |.|
    import :mod:`ontology.OntologyUtils`

************************

'''

__author__ = "cued_dialogue_systems_group"
import ontology.FlatOntologyManager, os, copy
from ontology import OntologyUtils, DataBaseSQLite
from utils import ContextLogger, Settings
from cedm.utils import DActEntity
logger = ContextLogger.getLogger('')



#------------------------------------------------------------------------------------------------------------
# ONTOLOGY FOR A SINGLE DOMAIN
#------------------------------------------------------------------------------------------------------------    
class FlatEntityOntologyManager(ontology.FlatOntologyManager.FlatOntologyManager):
    def __init__(self):
        super(FlatEntityOntologyManager, self).__init__()
        
    def get_common_slots(self, eType1, eType2):
        rel_slots = []
        for slot1 in self.get_informable_slots(eType1):
            if slot1=='name':
                continue
            for slot2 in self.get_informable_slots(eType2):
                if slot1 == slot2:
                    # this should be sufficient (ie no entity ids necessary) as the slot is part of a 
                    # relation which is defined between two entities
                    rel_slots.append(slot1+'#'+slot2)
                     
#                     rel_slots.append(eType1+'#'+slot1+'#'+eType2+'#'+slot2)
        return rel_slots
    
    
    def get_common_slots_for_all_types(self, eType1):
        relations = {}
        for domain in self.possible_domains:
            if domain == eType1:
                continue
            relations[domain] = self.get_common_slots(eType1, domain)
        return relations
            
    def get_len_informable_slot(self, dstring, slot):
        if '#' in slot:
            # assume we are dealing with a relation which currently is hardcoded to one value
            return 1
        return len(self.ontologyManagers[dstring].ontology['informable'][slot])
    
    def _load_domains_ontology(self, domainString):
        '''
        Loads and instantiates the respective ontology object as configured in config file. The new object is added to the internal
        dictionary. 
        
        Default is FlatDomainOntology.
        
        .. Note:
            To dynamically load a class, the __init__() must take one argument: domainString.
        
        :param domainString: the domain the ontology will be loaded for.
        :type domainString: str
        :returns: None
        '''
        
        ontologyClass = None
        
        if Settings.config.has_option('ontology_' + domainString, 'handler'):
            ontologyClass = Settings.config.get('ontology_' + domainString, 'handler')
        
        if ontologyClass is None:
            return FlatEntityOntology(domainString)
        else:
            try:
                # try to view the config string as a complete module path to the class to be instantiated
                components = ontologyClass.split('.')
                packageString = '.'.join(components[:-1]) 
                classString = components[-1]
                mod = __import__(packageString, fromlist=[classString])
                klass = getattr(mod, classString)
                return klass(domainString)
            except ImportError:
                logger.error('Unknown domain ontology class "{}" for domain "{}"'.format(ontologyClass, domainString))
                
                
class FlatEntityOntology(ontology.FlatOntologyManager.FlatDomainOntology):
    def _set_db(self):
        """Sets self.db to instance of choosen Data base accessing class. 
        
        .. note:: It is currently hardcoded to use the sqlite method. But this can be config based - data base classes share interface
        so only need to change class here, nothing else in code will need adjusting. 
        """
        db_fname = OntologyUtils.get_database_path(self.domainString)
        logger.info('Loading database: '+db_fname+'db')
        try:
            #self.db = DataBase.DataBase(db_fname+'txt')
            dbprefix = None
            if Settings.config.has_option("exec_config", "dbprefix"):
                dbprefix = Settings.config.get("exec_config", "dbprefix")
                if dbprefix.lower() == 'none':
                    dbprefix = None
            if dbprefix:
                db_fname = os.path.join(dbprefix, db_fname.split('/')[-1])
            self.db = DataBaseEntitySQLite(dbfile=db_fname+'db', dstring=self.domainString)
        except IOError:
            print IOError
            logger.error("No such file or directory: "+db_fname+". Probably <Settings.root> is not set/set wrong by config.")
        return
    
class DataBaseEntitySQLite(DataBaseSQLite.DataBase_SQLite):
    def entity_by_features(self, constraints):
        '''Retrieves from database all entities matching the given constraints. 
       
        :param constraints: features. Dict {slot:value, ...} or List [(slot, op, value), ...] \
        (NB. the tuples in the list are actually a :class:`dact` instances)
        :returns: (list) all entities (each a dict)  matching the given features.
        '''
        
        # 1. Format constraints into sql_query 
        # NO safety checking - constraints should be a list or a dict 
        # Also no checking of values regarding none:   if const.val == [None, '**NONE**']: --> ERROR
        doRand = False
        
        
        if len(constraints):
            bits = []
            values = []
            if isinstance(constraints, list):
                constraints = copy.deepcopy(constraints)     
                # first remove relations from constraints (should only happen when evaluating and using user goal for querying)
                for i in xrange(len(constraints)-1,-1,-1): # counting down from len(constraints)-1 (included) to 0 (-1 is excluded) in steps of 1
                    const = constraints[i]
                    if isinstance(const, DActEntity.DactItemEntity) and const.is_relation():
                        del constraints[i]
                
                for const in constraints:
                    if const.op == '=' and const.val == 'dontcare':
                        continue       # NB assume no != 'dontcare' case occurs - so not handling
                    if const.op == '!=' and const.val != 'dontcare':
                        bits.append(const.slot +'!= ?')
                    else:
                        bits.append(const.slot +'= ?  COLLATE NOCASE')
                    values.append(const.val)
            elif isinstance(constraints, dict):
                for slot,value in constraints.iteritems():
                    if value != 'dontcare':
                        bits.append(slot +'= ? COLLATE NOCASE')
                        values.append(value)
                        
            # 2. Finalise and Execute sql_query
            try:
                if len(bits):
                    sql_query = '''select  * 
                    from {} 
                    where '''.format(self.domain)
                    sql_query += ' and '.join(bits)
                    self.cursor.execute(sql_query, tuple(values))
                else:
                    sql_query =  self.no_constraints_sql_query
                    self.cursor.execute(sql_query)
                    doRand = True
            except Exception as e:
                print e     # hold to debug here
                logger.error('sql error ' + str(e))
                
                
            
        else:
            # NO CONSTRAINTS --> get all entities in database?  
            #TODO check when this occurs ... is it better to return a single, random entity? --> returning random 10
            
            # 2. Finalise and Execute sql_query
            sql_query =  self.no_constraints_sql_query
            self.cursor.execute(sql_query)
            doRand = True
        
        results = self.cursor.fetchall()        # can return directly
        
        if doRand:
            Settings.random.shuffle(results)
        return results

#END OF FILE
