# coding=utf-8
"""Modalities with Sequence level label smoothing loss function."""

import tensorflow as tf

from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry
from tensor2tensor.layers import common_layers
from tensor2tensor.layers.common_layers import weights_nonzero, FactoredTensor, shape_list, pad_with_zeros

from usr.utils import print_shape, print_data


def padded_cross_entropy_seqls(logits,
                         labels,
                         label_smoothing,
                         weights_fn=weights_nonzero,
                         reduce_sum=True,
                         cutoff=0.0,
                         gaussian=False):
  """Compute cross-entropy assuming 0s are padding.

  Computes a loss numerator (the sum of losses), and loss denominator
  (the number of non-padding tokens).

  Args:
    logits: a `Tensor` with shape `[batch, timesteps, vocab_size]`.
      optionally a FactoredTensor.
    labels: an integer `Tensor` with shape `[batch, timesteps]`.
    label_smoothing: a floating point `Scalar`.
    weights_fn: A function from labels to weights.
    reduce_sum: a Boolean, whether to sum at the end or not.
    cutoff: a float, at which point to have no loss.
    gaussian: If true, use a Gaussian distribution for label smoothing

  Returns:
    loss_numerator: a `Scalar`.  Sum of losses.
    loss_denominator: a `Scalar.  The number of non-padding target tokens.

  Raises:
    ValueError: in case of unsupported argument types.
  """
  if isinstance(logits, FactoredTensor) or gaussian:
    raise ValueError("Gaussian smoothing not implemented because it's BS. "
                     "Factored loss not implemented yet.")
  confidence = 1.0 - label_smoothing
  logits_shape = shape_list(logits)
  vocab_size = logits_shape[-1]
  with tf.name_scope("padded_cross_entropy", values=[logits, labels]):
    logits, labels = pad_with_zeros(logits, labels)
    logits = tf.reshape(
        logits,
        shape_list(labels) + [vocab_size],
        name="padded_cross_entropy_size_check")
    logits = tf.cast(logits, tf.float32)
    weights = weights_fn(labels)
    xent = smoothing_cross_entropy_seqls(
        logits, labels, vocab_size, confidence, weights=weights)
    if cutoff > 0.0:
      xent = tf.nn.relu(xent - cutoff)
    if not reduce_sum:
      return xent * weights, weights
    return tf.reduce_sum(xent * weights), tf.reduce_sum(weights)


def smoothing_cross_entropy_seqls(logits,
                            labels,
                            vocab_size,
                            confidence,
                            weights,
                            normalize=False):
  """Cross entropy with label smoothing to limit over-confidence.

  Args:
    logits: Tensor of shape [batch_size, ?, ?, ?, vocab_size].
    labels: Tensor of shape [batch_size, ?, ?, ?].
    vocab_size: Tensor representing the size of the vocabulary.
    confidence: Used to determine on and off values for label smoothing.
      If `gaussian` is true, `confidence` is the variance to the Gaussian
      distribution.

  Returns:
    Tensor of shape [batch_size, ?, ?, ?].
  """
  with tf.name_scope("smoothing_cross_entropy", values=[logits, labels]):
    seq_lengths = tf.reduce_sum(weights, axis=1)
    seq_confidence = tf.pow(confidence, 1.0 / seq_lengths)
    
    # Low confidence is given to all non-true labels, uniformly.
    seq_low_confidence = (1.0 - seq_confidence) / tf.to_float(vocab_size - 1)

    soft_targets = tf.one_hot(
        tf.cast(labels, tf.int32),
        depth=vocab_size,
        on_value=1.0,
        off_value=0.0)
    soft_targets *= tf.expand_dims(tf.expand_dims(seq_confidence - seq_low_confidence, -1), -1)
    soft_targets += tf.expand_dims(tf.expand_dims(seq_low_confidence, -1), -1)
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=soft_targets)
    if normalize:
      seq_normalizing = -(
          confidence * tf.log(seq_confidence) + tf.to_float(vocab_size - 1) *
          seq_low_confidence * tf.log(seq_low_confidence + 1e-20))
      return xentropy - tf.expand_dims(seq_normalizing, 1)
    return xentropy


@registry.register_symbol_modality("seqls")
class SymbolModalitySequuenceLabelSmoothing(modalities.SymbolModality):


  def loss(self, top_out, targets, weights_fn=None):
    """Compute loss numerator and denominator for one shard of output."""
    logits = top_out
    if weights_fn is None:
      weights_fn = self.targets_weights_fn
    return padded_cross_entropy_seqls(
        logits,
        targets,
        self._model_hparams.label_smoothing,
        weights_fn=weights_fn)
