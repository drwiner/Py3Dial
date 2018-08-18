Policy
*********************

.. |.| raw:: html

   <br />
   
.. automodule:: policy.Policy
    :members:
    :private-members:
    
.. automodule:: policy.PolicyManager
    :members:
    :private-members:
    
.. automodule:: policy.PolicyCommittee
    :members:
    :private-members:
         
.. automodule:: policy.HDCPolicy
.. autoclass:: policy.HDCPolicy.HDCPolicy
   
.. automodule:: policy.GPPolicy
.. autoclass:: policy.GPPolicy.GPPolicy
.. autoclass:: policy.GPPolicy.Kernel
.. autoclass:: policy.GPPolicy.GPAction
.. autoclass:: policy.GPPolicy.GPState
.. autoclass:: policy.GPPolicy.TerminalGPAction
.. autoclass:: policy.GPPolicy.TerminalGPState

.. automodule:: policy.GPLib
.. autoclass:: policy.GPLib.GPSARSA
.. autoclass:: policy.GPLib.GPSARSAPrior
.. autoclass:: policy.GPLib.LearnerInterface

.. automodule:: policy.HDCTopicManager
.. autoclass:: policy.HDCTopicManager.HDCTopicManagerPolicy

.. automodule:: policy.WikipediaTools
.. autoclass:: policy.WikipediaTools.WikipediaDM

.. automodule:: policy.SummaryAction
.. autoclass:: policy.SummaryAction.SummaryAction

.. automodule:: policy.SummaryUtils
 
.. automodule:: policy.PolicyUtils 
   :members:

.. automodule:: policy.BCM_Tools

DeepRL Policies
*********************

.. automodule:: policy.A2CPolicy

.. automodule:: policy.ACERPolicy

.. automodule:: policy.BDQNPolicy

.. automodule:: policy.DQNPolicy

.. automodule:: policy.ENACPolicy

.. automodule:: policy.TRACERPolicy

FeudalRL Policies
*********************

Traditional Reinforcement Learning algorithms fail to scale to large domains due to the curse of dimensionality. A novel Dialogue Management architecture based on Feudal RL decomposes the decision into two steps; a first step where a master policy selects a subset of primitive actions, and a second step where a primitive action is chosen from the selected subset. The structural information included in the domain ontology is used to abstract the dialogue state space, taking the decisions at each step using different
parts of the abstracted state. This, combined with an information sharing mechanism between slots, increases the scalability to large
domains.

For more information, please look at the paper `Feudal Reinforcement Learning for Dialogue Management in Large Domains <https://arxiv.org/pdf/1803.03232.pdf>`_.
 
 
 