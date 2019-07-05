# coding=utf-8
"""Base hyper parameter sets for TF-NMT."""

from tensor2tensor.layers import common_hparams

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
  hparams.shared_embedding_and_softmax_weights = False
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
  hparams.symbol_modality_num_shards = 16  # backwards compatibility
  return hparams


def tfnmt_default():
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


def convert_to_tfnmt_hparams(hparams):
  """Add hyper parameters required by TF-NMT but which are not directly
  accessible via T2T. This method extends the T2T hparams for using 
  them in the TF-NMT subpackage.
  """
  try:
    hparams.add_hparam("num_layers", hparams.num_hidden_layers)
  except ValueError:
    # A value error occurs when hparams.num_layers already exists, for example
    #  when using multiple GPUs. In this case we assume that hparams is
    # already converted.
    return hparams
  hparams.add_hparam("src_vocab_size", None)  # Not used
  hparams.add_hparam("tgt_vocab_size", None)  # Not used 
  hparams.add_hparam("num_gpus", 1)  # Not used
  hparams.add_hparam("time_major", False)  # True in TF-NMT 
  hparams.add_hparam("init_weight", 0.1 * hparams.initializer_gain)
  hparams.add_hparam("random_seed", None) 
  hparams.add_hparam("num_units", hparams.hidden_size) 
  hparams.add_hparam("sos", "foobar")  # Not used 
  hparams.add_hparam("eos", "foobar")  # Not used 
  hparams.add_hparam("tgt_max_len_infer", None)  # Not used 
  hparams.add_hparam("beam_width", 1)  # Not used 
  # See nmt.nmt.extend_hparams()
  # Sanity checks
  if hparams.encoder_type == "bi" and hparams.num_layers % 2 != 0:
    raise ValueError("For bi, num_layers %d should be even" %
                     hparams.num_layers)
  if (hparams.attention_architecture in ["gnmt"] and
      hparams.num_layers < 2):
    raise ValueError("For gnmt attention architecture, "
                     "num_layers %d should be >= 2" % hparams.num_layers)
  # Set num_residual_layers
  if hparams.residual and hparams.num_layers > 1:
    if hparams.encoder_type == "gnmt":
      # The first unidirectional layer (after the bi-directional layer) in
      # the GNMT encoder can't have residual connection since the input is
      # the concatenation of fw_cell and bw_cell's outputs.
      num_residual_layers = hparams.num_layers - 2
    else:
      # Last layer cannot have residual connections since the decoder
      # expects num_unit dimensional input
      num_residual_layers = hparams.num_layers - 1
  else:
    num_residual_layers = 0
  hparams.add_hparam("num_residual_layers", num_residual_layers)
  return hparams
