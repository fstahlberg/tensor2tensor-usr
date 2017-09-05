# coding=utf-8

import os
import sys

try:
  # TF-NMT
  from cam.models.tfnmt import tfnmt_model
  from cam.models.tfnmt import tfnmt_hparams
except ImportError as e:
  sys.exit("Import error: %s. Double-check that %s is in your PYTHONPATH"
           % (e, os.path.dirname(os.path.realpath(__file__))))

