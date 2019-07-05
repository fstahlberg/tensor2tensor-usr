# coding=utf-8

import os
import sys

try:
  # TF-NMT (https://github.com/tensorflow/nmt)
  import usr.models.tfnmt.model
  import usr.models.tfnmt.hparams

  # Layer-by-layer models
  #import usr.models.layerbylayer.tfnmt_model
  #import usr.models.layerbylayer.transformer_model
  #import usr.models.layerbylayer.hparams
  #import usr.models.layerbylayer.problem
  #import usr.models.layerbylayer.modalities

  # Ensembles
  import usr.models.ensemble

  # Glue models
  import usr.models.glue

  # Gibbs models
  import usr.models.gibbs

  # Context transformer
  import usr.models.transformer_context
  
  # Additional hparams and problems for standard T2T models
  import usr.configs.hparams
  import usr.configs.problem
  import usr.configs.problem_lt
  import usr.configs.problem_sn

  # Additional losses/modalities
  #import usr.modalities.gradout
  #import usr.modalities.seqls
  #import usr.modalities.simplefusion
except ImportError as e:
  sys.exit("Import error: %s. Double-check that %s is in your PYTHONPATH"
           % (e, os.path.dirname(os.path.realpath(__file__))))

