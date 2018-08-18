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
Created on 15 May 2017

@author: su259
'''
class Entity(object):
    '''
    classdocs
    '''
    def __init__(self, _type):
        self._type = _type
        self._belief = None
        
        
        
        
class Relation(object):
    
    def __init__(self, e1, e2, attribE1, attribE2, relation):
        self.e1 = e1
        self.e2 = e2
        self.attribE1 = attribE1
        self.attribE2 = attribE2
        self.relation1to2 = relation
        
    
        