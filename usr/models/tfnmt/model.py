# coding=utf-8
"""TF-NMT models for T2T.

This class contains an adaptor model implementation for using models from the
TensorFlow NMT tutorial

https://github.com/tensorflow/nmt
"""


from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.layers import common_layers
from usr import utils as usr_utils
from usr.models.tfnmt.nmt.nmt import attention_model
from usr.models.tfnmt.nmt.nmt import gnmt_model
from usr.models.tfnmt.nmt.nmt import model as nmt_model
from usr.models.tfnmt.nmt.nmt.utils import iterator_utils
from usr.models.tfnmt import hparams_helper

import tensorflow as tf


@registry.register_model
class TFNmt(t2t_model.T2TModel):
  """Adaptor class for TF-NMT models."""

  def model_fn_body(self, features):
    hparams = self._hparams
    inputs, inputs_length = usr_utils.get_feature_with_length(features, 
                                                              "inputs")
    targets, targets_length = usr_utils.get_feature_with_length(features, 
                                                                "targets")
    # We need to do +1 for inference since get_feature_with_length()
    # may not have direct access to sequence lengths and returns
    # a length of 0 for the first inference step. 
    if hparams.mode == tf.contrib.learn.ModeKeys.INFER:
      targets_length = targets_length + 1
    # inputs_length of 0 breaks things
    inputs_length = tf.maximum(inputs_length, 1)
    tfnmt_model = get_tfnmt_model(
        hparams, inputs, inputs_length, targets, targets_length)
    decoder_output = tfnmt_model.logits
    return tf.expand_dims(decoder_output, axis=2)




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
      hparams_helper.convert_to_tfnmt_hparams(hparams),
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
      target_input=common_layers.shift_right_3d(targets),
      target_output=None, # Loss is computed in T2T
      source_sequence_length=inputs_length,
      target_sequence_length=targets_length)

