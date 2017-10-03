# coding=utf-8

import os
import sys

#try:
if True:
  # TF-NMT
  import usr.models.tfnmt.model
  import usr.models.tfnmt.hparams
  
  # Layer-by-layer models
  import usr.models.layerbylayer.model
  import usr.models.layerbylayer.hparams
  import usr.models.layerbylayer.problem
  
  # Additional hparams and problems for standard T2T models
  import usr.configs.hparams
  import usr.configs.problem
#except ImportError as e:
#  sys.exit("Import error: %s. Double-check that %s is in your PYTHONPATH"
#           % (e, os.path.dirname(os.path.realpath(__file__))))

