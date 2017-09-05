"""Hyper parameter sets for TF-NMT.

This file contains the standard hyperparameters in the TF-NMT tutorial.
"""

from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
from tensor2tensor.layers import common_hparams

@registry.register_hparams
def tfnmt_base():
  """TF-NMT base configuration.

  Note: the SGD learning rate schedule is not replicated exactly. TF-NMT uses
  `learning_rate` until `start_decay_steps`, and then multiplies the learning
  rate with `decay_factor` every `decay_steps` steps.

  T2T uses the inverse decay rate until `learning_rate_warmup_steps` and then
  applies the `noam` decay scheme.

  Following fields are not covered by this as T2T defines them somewhere else.
    "num_train_steps": 12000,
    "steps_per_external_eval": null,
    "steps_per_stats": 100,
  """
  hparams = common_hparams.basic_params1()
  hparams.batch_size = 4096  # Roughly equivalent to TF-NMT's batch_size=128
  hparams.dropout = 0.2
  hparams.learning_rate = 1.0
  hparams.clip_grad_norm = 5.0  # Called max_gradient_norm in TF-NMT
  hparams.optimizer = "SGD"  # sgd in TF-NMT
  hparams.learning_rate_decay_scheme = "noam" # See docstring
  hparams.max_input_seq_length = 50  # Called max_src_len* in TF-NMT
  hparams.max_target_seq_length = 50  # Called max_trg_len* in TF-NMT
  hparams.initializer = "uniform"
  hparams.initializer_gain = 1.0
  hparams.add_hparam("attention", "normed_bahdanau")
  hparams.add_hparam("attention_architecture", "standard")
  hparams.add_hparam("encoder_type", "bi")
  hparams.add_hparam("forget_bias", 1.0)
  hparams.add_hparam("unit_type", "lstm")
  hparams.add_hparam("residual", False)
  hparams.add_hparam("pass_hidden_state", True)
  return hparams


# The following hparams are taken from the nmt.standard_hparams directory in
# the TF NMT tutorial.


@registry.register_hparams
def tfnmt_iwslt15():
  """TF-NMT iwslt15 configuration.
  
  Following fields differ in the original setup:
    "decay_factor": 0.5,
    "decay_steps": 1000,
  """
  hparams = tfnmt_base()
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
  hparams = tfnmt_base()
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
  hparams = tfnmt_base()
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
  hparams = tfnmt_base()
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
  hparams = tfnmt_base()
  hparams.attention = "normed_bahdanau"
  hparams.encoder_type = "bi"
  hparams.num_hidden_layers = 4
  hparams.hidden_size = 1024
  hparams.residual = True
  hparams.unit_type = "layer_norm_lstm"
  hparams.learning_rate_warmup_steps = 4000
  hparams.learning_rate = 0.1
  hparams.optimizer = "Adam"
  hparams.optimizer_adam_epsilon = 1e-7
  hparams.optimizer_adam_beta1 = 0.85
  hparams.optimizer_adam_beta2 = 0.997
  return hparams

