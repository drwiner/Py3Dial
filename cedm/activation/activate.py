#!/bin/python

import os, shutil, sys

p, f = os.path.split(os.path.realpath(__file__))
root = os.path.abspath(os.path.join(p,'../../'))

src = os.path.join(p,"resources/SimulateCEDM.py")
dst = root
sys.stdout.write("Copying SimulateCEDM.py...")
sys.stdout.flush()
shutil.copy(src, dst)
print(" Done.")

src = os.path.join(p,"resources/AgentCEDM.py")
dst = root
sys.stdout.write("Copying AgentCEDM.py...")
sys.stdout.flush()
shutil.copy(src, dst)
print(" Done.")

src = os.path.join(p,"resources/TownInfo-dbase.db")
dst = os.path.join(root,"ontology/ontologies")
sys.stdout.write("Copying TownInfo-dbase.db...")
sys.stdout.flush()
shutil.copy(src, dst)
print(" Done.")

src = os.path.join(p,"resources/TownInfo-rules.json")
dst = os.path.join(root,"ontology/ontologies")
sys.stdout.write("Copying TownInfo-dbase.json...")
sys.stdout.flush()
shutil.copy(src, dst)
print(" Done.")

src = os.path.join(p,"resources/cedm.cfg")
dst = os.path.join(root,"config")
sys.stdout.write("Copying cedm.cfg...")
sys.stdout.flush()
shutil.copy(src, dst)
print(" Done.")
