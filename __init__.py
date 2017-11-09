# coding=utf-8

import os
import sys

try:
  # TF-NMT (https://github.com/tensorflow/nmt)
  import usr.models.tfnmt.model
  import usr.models.tfnmt.hparams

  # Extensions to the transformer model
  # - Advanced positional encodings
  # - Recurrent layers
  import usr.models.transformer_ext.model
  import usr.models.transformer_ext.hparams
  
  # Layer-by-layer models
  import usr.models.layerbylayer.tfnmt_model
  import usr.models.layerbylayer.transformer_model
  import usr.models.layerbylayer.hparams
  import usr.models.layerbylayer.problem
  import usr.models.layerbylayer.modalities
  
  # Additional hparams and problems for standard T2T models
  import usr.configs.hparams
  import usr.configs.problem
except ImportError as e:
  sys.exit("Import error: %s. Double-check that %s is in your PYTHONPATH"
           % (e, os.path.dirname(os.path.realpath(__file__))))

