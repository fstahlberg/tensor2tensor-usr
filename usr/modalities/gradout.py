# coding=utf-8
"""Modalities with GradOut loss function."""

import tensorflow as tf

from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry
from tensor2tensor.layers import common_layers

from usr.utils import print_shape, print_data


@registry.register_symbol_modality("gradout")
class SymbolModalityGradOut(modalities.SymbolModality):
  """SymbolModality with GradOut loss.

  GradOut samples from the target distribution at each time step. If
  the sampled label matches the target label, we mask the gradient
  (i.e. set it to zero) at that position, otherwise we apply normal
  cross-entropy loss.

  GradOut tries to focus training on symbols which have not yet been
  learned properly by the model. It can also mediate training issues
  when a specific symbol is over-represented in the training data
  (such as <self> in text norm) as gradients will be more sensitive to
  non-trivial cases once the model has learned to predict the trivial
  one.
  """

  def loss(self, top_out, targets, weights_fn=None):
    """Compute loss numerator and denominator for one shard of output."""
    logits = top_out
    shp = tf.shape(logits)
    sampled = tf.multinomial(
        tf.reshape(logits, [-1, shp[4]]), 1, output_dtype=tf.int32)
    sampled = tf.reshape(sampled, tf.shape(targets))
    mask = tf.to_int32(tf.not_equal(targets, sampled))
    targets = targets * mask
    if weights_fn is None:
      weights_fn = self.targets_weights_fn
    return common_layers.padded_cross_entropy(
        logits,
        targets,
        self._model_hparams.label_smoothing,
        weights_fn=weights_fn)
