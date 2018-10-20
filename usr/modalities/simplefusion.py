# coding=utf-8
"""SimpleFusion modality following https://arxiv.org/pdf/1809.00125.pdf"""

import tensorflow as tf

from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry
from tensor2tensor.layers import common_layers

from usr.utils import print_shape, print_data

from six.moves import range # pylint: disable=redefined-builtin

def log_prob_from_logits(logits):
  """Softmax function."""
  return logits - tf.reduce_logsumexp(logits, keepdims=True)


@registry.register_symbol_modality("simplefusion")
class SymbolModalitySimpleFusion(modalities.SymbolModality):
  """SymbolModality with Simple Fusion combination scheme."""

  #def __init__(self, model_hparams, vocab_size=None):
  #  super(SymbolModalitySimpleFusion, self).__init__(model_hparams, vocab_size)
    
  def fusion_mode(self):
    try:
      mode = self._model_hparams.ensemble_fusion_mode
    except AttributeError:
      mode = "prenorm"
    if mode not in ["prenorm", "postnorm", "share_embeddings"]:
      raise AttributeError("Unknown ensemble_fusion_mode '%s'" % mode)
    return mode

  def _get_weights(self, hidden_dim=None, model_id=None):
    """Overrides super class function."""
    if hidden_dim is None:
      hidden_dim = self._body_input_depth
    if self.fusion_mode() == "share_embeddings":
      return tf.get_variable(
              "ens_weights_shared", [self._vocab_size, hidden_dim],
              initializer=tf.random_normal_initializer(0.0, hidden_dim**-0.5))
    shards = []
    if model_id is None:
      model_ids = range(len(self._model_hparams.ensemble_hidden_sizes))
    else:
      model_ids = [model_id]
    for model_id in model_ids:
      model_hidden_size = self._model_hparams.ensemble_hidden_sizes[model_id]
      var_name = "ens_weights_%d" % model_id
      shards.append(
        tf.get_variable(
          var_name, [self._vocab_size, model_hidden_size],
          initializer=tf.random_normal_initializer(0.0, model_hidden_size**-0.5)))
    if len(shards) == 1:
      return shards[0]
    ret = tf.concat(shards, 1)
    # Convert ret to tensor.
    if not tf.contrib.eager.in_eager_mode():
      ret = common_layers.convert_gradient_to_tensor(ret)
    return ret

  def top(self, body_output, _):
    """Generate logits.
    Args:
      body_output: A Tensor with shape [batch, p0, p1, body_input_depth]
    Returns:
      logits: A Tensor with shape  [batch, p0, p1, 1, vocab_size].
    """
    if self._model_hparams.symbol_modality_skip_top:
      return tf.expand_dims(body_output, 3)

    if self._model_hparams.shared_embedding_and_softmax_weights:
      scope_name = "shared"
      reuse = True
    else:
      scope_name = "softmax"
      reuse = False

    with tf.variable_scope(scope_name, reuse=reuse):
      body_output_shape = common_layers.shape_list(body_output)
      body_output = tf.reshape(body_output, [-1, body_output_shape[-1]])
      if self.fusion_mode() == "share_embeddings":
        var = self._get_weights(body_output_shape[-1])
        logits = tf.matmul(body_output, var, transpose_b=True)
      else:
        pos = 0
        logits = None
        for model_id, (hidden_size, enabled, trainable) in enumerate(zip(
                                self._model_hparams.ensemble_hidden_sizes, 
                                self._model_hparams.ensemble_enabled,
                                self._model_hparams.ensemble_trainable)):
          projected = tf.matmul(body_output[:, pos:pos+hidden_size], 
                                self._get_weights(model_id=model_id), 
                                transpose_b=True)
          if not enabled:
            projected *= 0.0  # Disabled, but variables are still created
          if self.fusion_mode() == "postnorm":
            projected = log_prob_from_logits(projected)
          if not trainable:
            projected = tf.stop_gradient(projected)
          if logits is None:
            logits = projected
          else:
            logits += projected
          pos += hidden_size
      return tf.reshape(logits, body_output_shape[:-1] + [1, self._vocab_size])

