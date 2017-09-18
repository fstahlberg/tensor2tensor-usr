"""TF-NMT models for T2T.

This class contains an adaptor model implementation for using models from the
TensorFlow NMT tutorial

https://github.com/tensorflow/nmt

inside T2T.
"""


from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.layers import common_layers
from usr import utils as usr_utils
from usr.models.tfnmt.nmt.nmt import attention_model
from usr.models.tfnmt.nmt.nmt import gnmt_model
from usr.models.tfnmt.nmt.nmt import model as nmt_model
from usr.models.tfnmt.nmt.nmt.utils import iterator_utils

import tensorflow as tf


@registry.register_model
class TFNmt(t2t_model.T2TModel):
  """Adaptor class for TF-NMT models."""

  def model_fn_body(self, features):
    hparams = self._hparams
    inputs = features["inputs"]
    targets = features["targets"]
    # inputs/targets shape: (batch_size, [src|trg]_max_len, 1, embed_size)
    inputs = common_layers.flatten4d3d(inputs)
    targets = common_layers.flatten4d3d(targets)
    inputs_length = usr_utils.get_sequence_length(inputs)
    targets_length = usr_utils.get_sequence_length(targets)
    # We need to do +1 for inference since the get_sequence_length()
    # does not have direct access to sequence lengths and returns
    # a length of 0 for the first inference step. 
    if hparams.mode == tf.contrib.learn.ModeKeys.INFER:
      targets_length = targets_length + 1
    tfnmt_model = get_tfnmt_model(
        hparams, inputs, inputs_length, targets, targets_length)
    decoder_output = tfnmt_model.logits
    return tf.expand_dims(decoder_output, axis=2)


def _add_tfnmt_hparams(hparams):
  # Add hyper parameters required in TF-NMT
  hparams.add_hparam("num_layers", hparams.num_hidden_layers)
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
      # the GNMT encoder can't have residual connection due to the input is
      # the concatenation of fw_cell and bw_cell's outputs.
      num_residual_layers = hparams.num_layers - 2
    else:
      num_residual_layers = hparams.num_layers - 1
  else:
    num_residual_layers = 0
  hparams.add_hparam("num_residual_layers", num_residual_layers)
  return hparams


def get_tfnmt_model(hparams, inputs, inputs_length, targets, targets_length):
  """Adapted from nmt.train.train()."""
  if not hparams.attention:
    model_class = nmt_model.Model
  elif hparams.attention_architecture == "standard":
    model_class = attention_model.AttentionModel
  elif hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
    model_class = gnmt_model.GNMTModel
  else:
    raise ValueError("Unknown model architecture")
  tfnmt_model = model_class(
      _add_tfnmt_hparams(hparams),
      iterator=get_fake_iterator(
          inputs, inputs_length, targets, targets_length),
      mode=tf.contrib.learn.ModeKeys.EVAL,  # We use eval graph for training
      source_vocab_table=FakeVocabTable(),
      target_vocab_table=FakeVocabTable())
  return tfnmt_model


class FakeVocabTable(object):
  def lookup(self, unused_arg):
    return 99999999


def get_fake_iterator(inputs, inputs_length, targets, targets_length):
  return iterator_utils.BatchedInput(
      initializer=None,
      source=inputs,
      target_input=common_layers.shift_left_3d(targets),
      target_output=None, # Loss is computed in T2T
      source_sequence_length=inputs_length,
      target_sequence_length=targets_length)

