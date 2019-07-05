# coding=utf-8

"""Glue
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf
import copy

from usr import utils


@registry.register_model
class Glue(t2t_model.T2TModel):
  """Glued models.  See file docstring."""

  def __init__(self, *args, **kwargs):
    model_args = copy.deepcopy(args)
    model_kwargs = copy.deepcopy(kwargs)
    super(Glue, self).__init__(*args, **kwargs)
    try:
      self._glue_symbol = self._hparams.glue_symbol
    except:
      self._glue_symbol = 2
    if hasattr(model_args[0], "problem_hparams"):
      model_args[0].problem_hparams = None
    model_kwargs["problem_hparams"] = None
    model_class = registry.model(self._hparams.glue_model)
    if self._hparams.glue_is_lm:
      class ModelWithoutInput(model_class):
        # This hack is necessary because has_input() always returns
        # true if no problem hparams are provided
        @property
        def has_input(self):
          return False
      self._glue_model = ModelWithoutInput(*model_args, **model_kwargs)
    else:
      self._glue_model = model_class(*model_args, **model_kwargs)
  
  def body(self, features):
    glue_features = copy.copy(features)
    # Use the 'packed' dataset implementations
    if "inputs_seg" in glue_features:
      glue_features["inputs_segmentation"] = glue_features["inputs_seg"]
      glue_features["inputs_position"] = glue_features["inputs_pos"]
    glue_features["targets_segmentation"] = glue_features["targets_seg"]
    glue_features["targets_position"] = glue_features["targets_pos"]
    print(glue_features)
    return self._glue_model.body(glue_features)

