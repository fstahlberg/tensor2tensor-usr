# coding=utf-8

"""Model ensembles.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf
import copy


@registry.register_model
class Ensemble(t2t_model.T2TModel):
  """Model ensembles.  See file docstring."""

  def __init__(self, *args, **kwargs):
    org_args = copy.deepcopy(args)
    org_kwargs = copy.deepcopy(kwargs)
    super(Ensemble, self).__init__(*args, **kwargs)
    self._sanity_check()
    self._models = []
    for model_name, is_lm, hidden_size in zip(self._hparams.ensemble_models,
                                              self._hparams.ensemble_is_lm,
                                              self._hparams.ensemble_hidden_sizes):
      this_args = copy.deepcopy(org_args)
      this_kwargs = copy.deepcopy(org_kwargs)
      if hasattr(this_args[0], "problem_hparams"):
        this_args[0].problem_hparams = None
      this_kwargs["problem_hparams"] = None
      this_args[0].hidden_size = hidden_size
      model_class = registry.model(model_name)
      if is_lm:
        class ModelWithoutInput(model_class):
          # This hack is necessary because has_input() always returns
          # true if no problem hparams are provided
          @property
          def has_input(self):
            return False
        model = ModelWithoutInput(*this_args, **this_kwargs)
      else:
        model = model_class(*this_args, **this_kwargs)
      self._models.append(model)
  
  def _sanity_check(self):
    if self._hparams.ensemble_fusion_mode == "share_embeddings":
      if not all(s == self._hparams.hidden_size 
                 for s in self._hparams.ensemble_hidden_sizes):
        raise AttributeError(
           "All ensemble_hidden_sizes must be equal to hidden_size=%d if "
           "ensemble_fusion_mode=share_embeddings" % self._hparams.hidden_size)
    else:
      if sum(self._hparams.ensemble_hidden_sizes) != self._hparams.hidden_size:
        raise AttributeError(
           "The sum of ensemble_hidden_sizes must be equal to hidden_size=%d "
           "if ensemble_fusion_mode is not set to 'share_embeddings'."
           % self._hparams.hidden_size)

  def body(self, features):
    if self._hparams.ensemble_fusion_mode == "share_embeddings":    
      return self.body_shared_embeddings(features)
    has_inputs = "inputs" in features
    if has_inputs:
      inputs = features["inputs"]
    else:
      tf.logging.info("Ensemble does not have inputs.")
    targets = features["targets"]
    body_outs = []
    pos = 0
    for i, (model, is_lm, hidden_size) in enumerate(zip(
            self._models, 
            self._hparams.ensemble_is_lm,
            self._hparams.ensemble_hidden_sizes)):
      with tf.variable_scope("ens_model_%d" % i):
        if has_inputs:
          features["inputs"] = inputs[:, :, :, pos:pos+hidden_size] 
        features["targets"] = targets[:, :, :, pos:pos+hidden_size]
        body_outs.append(model.body(features))
        pos += hidden_size
    return tf.concat(body_outs, axis=-1)

  def body_shared_embeddings(self, features):
    inputs = features["inputs"]
    targets = features["targets"]
    body_outs = []
    for i, (model, enabled, trainable) in enumerate(zip(
            self._models, 
            self._hparams.ensemble_enabled,
            self._hparams.ensemble_trainable)):
      with tf.variable_scope("ens_model_%d" % i):
        body_out = model.body(features)
        if not enabled:
          body_out *= 0.0  # Disabled, but variables are still created
        if not trainable:
          body_out = tf.stop_gradient(body_out)
        body_outs.append(body_out)
    return sum(body_outs)
