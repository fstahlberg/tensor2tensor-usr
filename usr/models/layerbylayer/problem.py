# coding=utf-8
"""Problem specifications for layer-by-layer models."""

import os

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators.wmt import TranslateEndeWmt32k
from tensor2tensor.utils import registry
from tensor2tensor.data_generators.problem import preprocess_example_common

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
    # Annoyingly this is done for 'inputs' in problem.serving_input_fn
    # but not for other input modalities
    while len(example["target_roots"].get_shape()) < 3:
      example["target_roots"] = tf.expand_dims(example["target_roots"], axis=-1)
    return preprocess_example_common(example, hparams, mode)


  def hparams(self, defaults, model_hparams):
    p = defaults
    target_vocab_size = self._encoders["targets"].vocab_size
    p.target_modality = (registry.Modalities.SYMBOL, target_vocab_size)
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
          # This is a hack to get the ID of the POP symbol.
          try:
            pop_id = self._encoders["targets"]._subtoken_string_to_id[POP]
          except AttributeError:
            try:
              pop_id = self._encoders["targets"]._token_to_id[POP]
            except AttributeError as e:
              tf.logging.fatal("Could not extract ID for %s. Please either "
                               "use the Subword or Token encoder or specify "
                               "the POP ID directly with hparams.pop_id."
                               % POP)
              raise e
        except KeyError as e:
          tf.logging.fatal("%s could not be found in the target vocabulary! "
                          "Please either add %s to the vocabulary file or "
                          "specify the POP ID directly with hparams.pop_id"
                          % (POP, POP))
          raise e
        tf.logging.info("Use ID %d for pop signal %s", pop_id, POP)
        model_hparams.add_hparam("pop_id", pop_id)


@registry.register_problem
class TranslateLayerbylayerPopEndeWmt32k(AbstractLayerbylayerProblem, TranslateEndeWmt32k):
  pass


@registry.register_problem
class TranslateLayerbylayerEndeWmt32k(AbstractLayerbylayerProblem, TranslateEndeWmt32k):
  pass
