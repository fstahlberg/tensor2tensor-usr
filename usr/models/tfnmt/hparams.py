# coding=utf-8
"""Hyper parameter sets for TF-NMT.

This file contains the standard hyperparameters in the TF-NMT tutorial.
"""

from tensor2tensor.utils import registry
from tensor2tensor.layers import common_hparams
from usr.models.tfnmt import hparams_helper


# The following hparams are taken from the nmt.standard_hparams directory in
# the TF NMT tutorial.


@registry.register_hparams
def tfnmt_iwslt15():
  """TF-NMT iwslt15 configuration.
  
  Following fields differ in the original setup:
    "decay_factor": 0.5,
    "decay_steps": 1000,
  """
  hparams = hparams_helper.tfnmt_base()
  hparams.attention = "scaled_luong"
  hparams.num_hidden_layers = 2  # Called num_layers in TF-NMT
  hparams.hidden_size = 512  # Called num_units in TF-NMT
  hparams.learning_rate_warmup_steps = 8000  # start_decay_step in TF-NMT
  return hparams


@registry.register_hparams
def tfnmt_wmt16():
  """TF-NMT wmt16 configuration.
  
  Following fields differ in the original setup:
    "decay_factor": 0.5,
    "decay_steps": 17000,
  """
  hparams = hparams_helper.tfnmt_base()
  hparams.attention = "normed_bahdanau"
  hparams.num_hidden_layers = 4  # Called num_layers in TF-NMT
  hparams.hidden_size = 1024  # Called num_units in TF-NMT
  hparams.learning_rate_warmup_steps = 170000  # start_decay_step in TF-NMT
  return hparams


@registry.register_hparams
def tfnmt_wmt16_gnmt_4_layer():
  """TF-NMT wmt16_gnmt_4_layer configuration.
  
  Following fields differ in the original setup:
    "decay_factor": 0.5,
    "decay_steps": 17000,
  """
  hparams = hparams_helper.tfnmt_base()
  hparams.attention = "normed_bahdanau"
  hparams.attention_architecture = "gnmt_v2"
  hparams.encoder_type = "gnmt"
  hparams.num_hidden_layers = 4  # Called num_layers in TF-NMT
  hparams.hidden_size = 1024  # Called num_units in TF-NMT
  hparams.residual = True
  hparams.learning_rate_warmup_steps = 170000  # start_decay_step in TF-NMT
  return hparams


@registry.register_hparams
def tfnmt_wmt16_gnmt_8_layer():
  """GNMT wmt16_gnmt_8_layer configuration.
  
  Following fields differ in the original setup:
    "decay_factor": 0.5,
    "decay_steps": 17000,
  """
  hparams = hparams_helper.tfnmt_base()
  hparams.attention = "normed_bahdanau"
  hparams.attention_architecture = "gnmt_v2"
  hparams.encoder_type = "gnmt"
  hparams.num_hidden_layers = 8  # Called num_layers in TF-NMT
  hparams.hidden_size = 1024  # Called num_units in TF-NMT
  hparams.residual = True
  hparams.learning_rate_warmup_steps = 170000  # start_decay_step in TF-NMT
  return hparams


@registry.register_hparams
def tfnmt_wmt16_gnmt_8_layer_adam():
  """GNMT wmt16_gnmt_8_layer configuration with Adam."""
  hparams = tfnmt_wmt16_gnmt_8_layer()
  hparams.attention = "normed_bahdanau"
  hparams.attention_architecture = "gnmt_v2"
  hparams.encoder_type = "gnmt"
  hparams.num_hidden_layers = 8  # Called num_layers in TF-NMT
  hparams.hidden_size = 1024  # Called num_units in TF-NMT
  hparams.residual = True
  hparams.learning_rate_warmup_steps = 4000
  hparams.learning_rate = 0.1
  hparams.optimizer = "Adam"
  hparams.optimizer_adam_epsilon = 1e-7
  hparams.optimizer_adam_beta1 = 0.85
  hparams.optimizer_adam_beta2 = 0.997
  return hparams


@registry.register_hparams
def tfnmt_wmt16_gnmt_8_layer_layer_norm():
  """GNMT wmt16_gnmt_8_layer configuration with Adam."""
  hparams = tfnmt_wmt16_gnmt_8_layer()
  hparams.unit_type = "layer_norm_lstm"
  return hparams


@registry.register_hparams
def tfnmt_wmt16_gnmt_8_layer_adam_layer_norm():
  """GNMT wmt16_gnmt_8_layer configuration with Adam."""
  hparams = tfnmt_wmt16_gnmt_8_layer_adam()
  hparams.unit_type = "layer_norm_lstm"
  return hparams


@registry.register_hparams
def tfnmt_wmt17_uedinish():
  """Similar to the stacked architecture in the WMT17 UEdin submission.
  
  Differs from the evaluation system as follows:
    - No backtranslation
    - LSTM instead of GRU
  """
  hparams = hparams_helper.tfnmt_default()
  return hparams


@registry.register_hparams
def tfnmt_12gb_gpu():
  """Inspired by WMT17 UEdin submission, but with a different training
  setup.
  """
  hparams = hparams_helper.tfnmt_default()
  hparams.learning_rate_warmup_steps = 6000
  hparams.label_smoothing = 0.1
  hparams.batch_size = 8192
  return hparams


@registry.register_hparams
def tfnmt_12gb_gpu_alternating():
  """tfnmt_12gb_gpu with alternating encoder."""
  hparams = tfnmt_12gb_gpu()
  hparams.residual = True
  hparams.encoder_type = "alternating"
  hparams.batch_size = 4096
  return hparams
