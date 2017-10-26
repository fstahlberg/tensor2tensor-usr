# coding=utf-8
"""Problem specifications for layer-by-layer models."""

import os

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators.wmt import TranslateEndeWmt32k
from tensor2tensor.utils import registry
from tensor2tensor.data_generators.problem import preprocess_example_common
from tensor2tensor.data_generators.problem import SpaceID
from usr import utils as usr_utils

import tensorflow as tf


POP = "##POP##"
"""Textual representation of the POP symbol."""


class AbstractLayerbylayerProblem(object):
  """
  We do not inherit from `Problem` to minimalize confusion with
  multiple inheritance.
  """

  def generator(self, data_dir, tmp_dir, train):
    """Data generation outside of T2T."""
    raise NotImplementedError

  def feature_encoders(self, data_dir):
    """Implementation of feature_encoders() for layerbylayer problems."""
    vocab_filename = os.path.join(data_dir, "vocab.%s" % self.name)
    if self.is_character_level:
      encoder = text_encoder.ByteTextEncoder()
    elif self.use_subword_tokenizer:
      encoder = text_encoder.SubwordTextEncoder(vocab_filename)
    else:
      encoder = text_encoder.TokenTextEncoder(vocab_filename)
    if self.has_inputs:
      return {"inputs": encoder, "targets": encoder, "target_roots": encoder}
    return {"targets": encoder, "target_roots": encoder}

  def example_reading_spec(self):
    data_fields = {
        "inputs": tf.VarLenFeature(tf.int64),
        "targets": tf.VarLenFeature(tf.int64),
        "target_roots": tf.VarLenFeature(tf.int64)
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)

  def preprocess_example(self, example, mode, hparams):
    # Create loss mask if necessary
    if hparams.use_loss_mask and mode == tf.estimator.ModeKeys.TRAIN:
      indices = tf.cumsum(tf.cast(tf.equal(example["targets"], hparams.pop_id),
                                  tf.int32))
      expanded_target_roots = usr_utils.expand_memory_by_pop_1d(
          tf.equal(example["targets"], hparams.pop_id), example["target_roots"])
      expanded_target_roots = expanded_target_roots[:-1]  # Compensate for offset
      example["loss_mask"] = tf.cast(tf.logical_or(
              tf.equal(expanded_target_roots, text_encoder.EOS_ID),
              tf.greater(expanded_target_roots, hparams.max_terminal_id),
          ), tf.int32)
      while len(example["loss_mask"].get_shape()) < 3:
        example["loss_mask"] = tf.expand_dims(example["loss_mask"], axis=-1)
    # Annoyingly this is done for 'inputs' in problem.serving_input_fn
    # but not for other input modalities
    while len(example["target_roots"].get_shape()) < 3:
      example["target_roots"] = tf.expand_dims(example["target_roots"], axis=-1)
    return preprocess_example_common(example, hparams, mode)


  def hparams(self, defaults, model_hparams):
    p = defaults
    target_vocab_size = self._encoders["targets"].vocab_size
    p.target_modality = (registry.Modalities.SYMBOL, target_vocab_size)
    if model_hparams.shared_embedding_and_softmax_weights:
      p.input_modality = {
          "target_roots": (registry.Modalities.SYMBOL, target_vocab_size)
      }
    else: # Keep target root embeddings separate
      p.input_modality = {
          "target_roots": ("%s:targetroots" % registry.Modalities.SYMBOL, 
                           target_vocab_size)
      }
    if self.has_inputs:
      source_vocab_size = self._encoders["inputs"].vocab_size
      p.input_modality["inputs"] = (registry.Modalities.SYMBOL, source_vocab_size)
      p.input_space_id = self.input_space_id
    p.target_space_id = self.target_space_id
    if self.is_character_level:
      p.loss_multiplier = 2.0
    # Lookup POP id.
    if model_hparams.target_root_attention == "pop":
      if not hasattr(model_hparams, 'pop_id'):
        try:
          pop_id = _get_string_to_id(self._encoders["targets"])[POP]
        except KeyError as e:
          tf.logging.fatal("%s could not be found in the target vocabulary! "
                          "Please either add %s to the vocabulary file or "
                          "specify the POP ID directly with hparams.pop_id"
                          % (POP, POP))
          raise e
        tf.logging.info("Use ID %d for pop signal %s", pop_id, POP)
        model_hparams.add_hparam("pop_id", pop_id)
    # Extract maximum terminal ID
    if model_hparams.use_loss_mask:
      if not hasattr(model_hparams, 'max_terminal_id'):
        string_to_id = _get_string_to_id(self._encoders["target_roots"])
        max_terminal_id = 0
        min_nonterminal_id = 1000000
        for s, i in string_to_id.iteritems():
          if s[:2] == "##" and s[-2:] == "##":
            min_nonterminal_id = min(min_nonterminal_id, i)
          else:
            max_terminal_id = max(max_terminal_id, i)
        if min_nonterminal_id != max_terminal_id + 1:
          tf.logging.warn("Overlapping non-terminal and terminal ID "
                          "ranges: min nonterminal is %d" % min_nonterminal_id)
        tf.logging.info("Set maximum terminal ID to %d", max_terminal_id)
        model_hparams.add_hparam("max_terminal_id", max_terminal_id)


def _get_string_to_id(encoder):
  """Get a dictionary from string representations to token ID for
  the encoder. This is a hack and only works for subword and token encoders.
  """
  try:
    return encoder._subtoken_string_to_id
  except AttributeError:
    try:
      return encoder._token_to_id
    except AttributeError as e:
      tf.logging.fatal("Could not read encoder vocabulary. Please either "
                        "use the Subword or Token encoder or specify "
                        "the POP ID and maximum terminal ID directly with "
                         "hparams.pop_id and hparams.max_terminal_id.")
      raise e

  
@registry.register_problem
class TranslateLayerbylayerPopEndeWmt32k(AbstractLayerbylayerProblem, TranslateEndeWmt32k):
  pass


@registry.register_problem
class TranslateLayerbylayerEndeWmt32k(AbstractLayerbylayerProblem, TranslateEndeWmt32k):
  pass


@registry.register_problem
class TranslateLayerbylayerPopPtb16k(AbstractLayerbylayerProblem, TranslateEndeWmt32k):
  @property
  def input_space_id(self):
    return SpaceID.EN_TOK

  @property
  def target_space_id(self):
    return SpaceID.EN_TOK

  @property
  def num_shards(self):
    return 1

  @property
  def vocab_name(self):
    return "vocab.wsj"

  @property
  def targeted_vocab_size(self):
    return 2**14  # 16k


@registry.register_problem
class TranslateLayerbylayerPtb16k(TranslateLayerbylayerPopPtb16k):
  pass
