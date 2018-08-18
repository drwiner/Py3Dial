Conversational Entity Dialogue Model
************************************

The Conversational Entity Dialogue Model (CEDM) aims at resolving restrictions of conventional single- or multi-domain dialogue models in their capabilities of modelling complex dialogue structures, e.g., relations. The CEDM is a novel dialogue model that is centred around entities and is able to model relations as well as multiple entities of the same type `(Ultes et al., 2018)
<http://www.pydial.org/cedm>`_. A prototype of the CEDM has been implemented and integrated into the `PyDial <http://www.pydial.org/>`_ toolkit.

Please note that the CEDM prototype implementation has been designed in a way to exploit structures and implementations of PyDial where possible. Thus, some restrictions needed to be posed on the implementation which need to be resolved in future versions.


Usage Instructions
------------------

The CEDM is disabled by default. To activate and use it, download the latest version of PyDial. From Pydial root, run the activation shell script 

.. code-block:: shell

   python cedm/activation/activate.py

which will copy all necessary files to their respective locations. You will now find a SimulateCEDM.py in the root folder of PyDial which you can use to run experiments with the CEDM. To test it, run 

.. code-block:: shell

   python SimulateCEDM.py -c config/cedm_example.cfg
