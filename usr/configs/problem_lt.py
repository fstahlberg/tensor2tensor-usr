# coding=utf-8
"""Additional problems for standard T2T models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators.translate_ende import TranslateEndeWmt32k
from tensor2tensor.data_generators.lm1b import LanguagemodelLm1b32k
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators.problem import preprocess_example_common
from usr import utils as usr_utils
import os

import tensorflow as tf
import numpy as np
import random


data_alpha = None
data_beta = None
desired_alpha = None
invert_ratio = None

def norm_pdf(x, mu=0.0, sigma=2.0):
  return 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))


def _filter_by_ratio_py(inputs, targets):
  global ratio_desired_mean
  u = random.random()
  ratio = float(targets.shape[0]) / float(inputs.shape[0])
  threshold = norm_pdf(ratio, ratio_desired_mean) / norm_pdf(ratio_desired_mean, ratio_desired_mean)
  #print("%f > %f (%f, %f)" % (threshold, u, ratio_desired_mean, ratio))
  return threshold > u

def filter_by_ratio(example):
  #return tf.shape(example["inputs"])[0] > tf.shape(example["targets"])
  return tf.py_func(_filter_by_ratio_py, [example["inputs"],example["targets"]], tf.bool)

def beta_pdf_unnorm_py(x, alpha, beta):
  return x**(alpha-1.0)*(1.0-x)**(beta-1.0)




def beta_pdf_unnorm(x, alpha, beta):
  return tf.pow(x, alpha-1.0)*tf.pow(1.0-x, beta-1.0)

def filter_rejection_sampling(example):
  global data_alpha, data_beta, desired_alpha, invert_ratio
  ilen = tf.to_float(tf.shape(example["inputs"])[0])
  tlen = tf.to_float(tf.shape(example["targets"])[0])
  ratio = tlen / (ilen + tlen)
  if invert_ratio:
    ratio = 1.0 - ratio
  threshold = beta_pdf_unnorm(ratio, desired_alpha, data_beta) / beta_pdf_unnorm(ratio, data_alpha, data_beta) 
  r = tf.random_uniform((1,))
  return (r < threshold)[0]

@registry.register_problem
class TranslateDeenWmt19BaseLt(TranslateEndeWmt32k):

  @property
  def src_vocab_file(self):
    return "vocab.ende.wmt19"
  @property
  def trg_vocab_file(self):
    return "vocab.ende.wmt19"

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

  def preprocess_example(self, example, mode, hparams):
    """Runtime preprocessing.

    Return a dict or a tf.Data.Datset.from_tensor_slices (if you want each
    example to turn into multiple).

    Args:
      example: dict, features
      mode: tf.estimator.ModeKeys
      hparams: HParams, model hyperparameters

    Returns:
      dict or Dataset
    """
    global data_alpha, data_beta, desired_alpha, invert_ratio
    example = preprocess_example_common(example, hparams, mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      #example["inputs"] = usr_utils.print_shape(example["inputs"], "inputs", dtype=tf.int64)
      tf.logging.info("Filter by length!")
      if hparams.lt_data_alpha < hparams.lt_desired_alpha: # Normal case: make longer
        data_alpha = hparams.lt_data_alpha
        data_beta = hparams.lt_data_beta
        desired_alpha = hparams.lt_desired_alpha
        invert_ratio = False
      else: # Make translation shorter
        tf.logging.info("Invert ratios (make translations shorter)")
        data_alpha =1.0 -  hparams.lt_data_alpha
        data_beta = hparams.lt_data_beta
        desired_alpha = 1.0 -hparams.lt_desired_alpha
        invert_ratio = True
      dataset = tf.data.Dataset.from_tensors(example)
      filtered = dataset.filter(filter_rejection_sampling)
      return filtered
    return example

