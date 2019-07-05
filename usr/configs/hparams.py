# coding=utf-8
"""Additional hparams for standard T2T models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from tensor2tensor.utils import registry
from tensor2tensor.models.transformer import transformer_base_v2, transformer_big, transformer_relative_big
from tensor2tensor.models.slicenet import slicenet_params1_noam, slicenet_params1

# BACKWARDS COMPATIBILITY

@registry.register_hparams("slicenet_1_shards16")
def slicenet_params1_shards16():
  hparams = slicenet_params1()
  hparams.symbol_modality_num_shards = 16
  return hparams


# CONTEXT TRANSFORMER

@registry.register_hparams
def transformer_context_base_v2_large_batch4():
  hparams = transformer_base_v2_large_batch4()
  hparams.add_hparam("transformer_context_types", ["inter", "intra"])
  return hparams

@registry.register_hparams
def transformer_context_big_large_batch4():
  hparams = transformer_big_large_batch2()
  hparams.optimizer_multistep_accumulate_steps = 4
  hparams.add_hparam("transformer_context_types", ["inter", "intra"])
  return hparams

@registry.register_hparams
def transformer_context_big_large_batch6():
  hparams = transformer_big_large_batch2()
  hparams.optimizer_multistep_accumulate_steps = 6
  hparams.batch_size = 3072
  hparams.add_hparam("transformer_context_types", ["inter", "intra"])
  return hparams

# SUBWORD NOISE

@registry.register_hparams
def transformer_base_v2_sn1_large_batch2():
  hparams = transformer_base_v2_large_batch2()
  hparams.add_hparam("sn_wmap", "/home/fs439/rds/hpc-work/data/wmt18/supplementary/wmap.bpe.ende")
  hparams.add_hparam("sn_bpe_codes", "/home/fs439/rds/hpc-work/data/wmt18/supplementary/bpe.train.ende")
  #hparams.add_hparam("sn_wmap", "/data/mifs_scratch/fs439/data/wmt19/supplementary/wmap.bpe.ende")
  #hparams.add_hparam("sn_bpe_codes", "/data/mifs_scratch/fs439/data/wmt19/supplementary/bpe.train.ende")
  hparams.add_hparam("sn_dropout", 0.1)
  return hparams

@registry.register_hparams
def transformer_base_v2_sn2_large_batch2():
  hparams = transformer_base_v2_sn1_large_batch2()
  hparams.sn_dropout = 0.2
  return hparams

@registry.register_hparams
def transformer_big_sn1_large_batch4():
  hparams = transformer_big_large_batch4()
  hparams.add_hparam("sn_wmap", "/home/fs439/rds/hpc-work/data/wmt18/supplementary/wmap.bpe.ende")
  hparams.add_hparam("sn_bpe_codes", "/home/fs439/rds/hpc-work/data/wmt18/supplementary/bpe.train.ende")
  #hparams.add_hparam("sn_wmap", "/data/mifs_scratch/fs439/data/wmt19/supplementary/wmap.bpe.ende")
  #hparams.add_hparam("sn_bpe_codes", "/data/mifs_scratch/fs439/data/wmt19/supplementary/bpe.train.ende")
  hparams.add_hparam("sn_dropout", 0.1)
  return hparams


# LENGTH TRAINING

@registry.register_hparams
def transformer_base_v2_lt5_deen_large_batch2():
  hparams = transformer_base_v2_large_batch2()
  # alpha=(4*0.488807)/(1.0-0.488807)
  hparams.add_hparam("lt_data_alpha", 3.824833)
  hparams.add_hparam("lt_data_beta", 4.0)
  # but we want a mean of 0.55 = 5.0/(4.0+5.0)
  hparams.add_hparam("lt_desired_alpha", 5.0)
  return hparams

@registry.register_hparams
def transformer_base_v2_lt7_deen_large_batch2():
  hparams = transformer_base_v2_large_batch2()
  # alpha=(4*0.488807)/(1.0-0.488807)
  hparams.add_hparam("lt_data_alpha", 3.824833)
  hparams.add_hparam("lt_data_beta", 4.0)
  # but we want a mean of 0.63 = 7.0/(4.0+7.0)
  hparams.add_hparam("lt_desired_alpha", 7.0)
  return hparams


# ENSEMBLES

@registry.register_hparams
def transformer_base_2ens_simplefusion():
  """Transformer with Simplefusion.
  
  Requires the T2T fork from https://github.com/fstahlberg/tensor2tensor
  """
  hparams = transformer_base_v2()
  hparams.target_modality = "symbol:simplefusion"
  hparams.input_modalities = "inputs:symbol:simplefusion"
  hparams.add_hparam("ensemble_fusion_mode", "share_embeddings") # prenorm, postnorm, share_embeddings
  hparams.add_hparam("ensemble_hidden_sizes", [512, 512])
  hparams.hidden_size = 512
  hparams.add_hparam("ensemble_enabled", [True, True])
  hparams.add_hparam("ensemble_trainable", [True, True])
  hparams.add_hparam("ensemble_is_lm", [False, False])
  hparams.add_hparam("ensemble_models", ["transformer", "transformer"])
  return hparams


@registry.register_hparams
def transformer_base_2ens_prenorm():
  hparams = transformer_base_2ens_simplefusion()
  hparams.ensemble_fusion_mode = "prenorm"
  hparams.ensemble_hidden_sizes = [512, 512]
  hparams.ensemble_enabled =  [True, True]
  hparams.ensemble_trainable = [False, True]
  hparams.hidden_size = 1024
  return hparams


@registry.register_hparams
def transformer_base_2ens_prenorm_large_batch2():
  hparams = transformer_base_2ens_prenorm()
  hparams.optimizer_multistep_accumulate_steps = 2
  hparams.optimizer = "MultistepAdam"
  return hparams


@registry.register_hparams
def transformer_base_tmlm_prenorm_large_batch2():
  hparams = transformer_base_2ens_prenorm_large_batch2()
  hparams.ensemble_is_lm = [True, False]
  return hparams


@registry.register_hparams
def transformer_base_lmlm_glue_prenorm_large_batch2():
  hparams = transformer_base_2ens_prenorm()
  hparams.ensemble_is_lm = [True, True]
  hparams.ensemble_models = ["glue", "transformer"]
  hparams.add_hparam("glue_model", "transformer")
  hparams.add_hparam("glue_is_lm", True)
  return hparams


@registry.register_hparams
def transformer_base_lmlm_glue_prenorm_large_batch4():
  hparams = transformer_base_lmlm_glue_prenorm_large_batch2()
  hparams.optimizer_multistep_accumulate_steps = 4
  return hparams





@registry.register_hparams
def transformer_base_2ens_postnorm_large_batch2():
  hparams = transformer_base_2ens_prenorm_large_batch2()
  hparams.ensemble_fusion_mode = "postnorm"
  return hparams


@registry.register_hparams
def transformer_base_tmlm_postnorm_large_batch2():
  hparams = transformer_base_tmlm_prenorm_large_batch2()
  hparams.ensemble_fusion_mode = "postnorm"
  return hparams

@registry.register_hparams
def transformer_base_lmlm_glue_postnorm_large_batch2():
  hparams = transformer_base_lmlm_glue_prenorm_large_batch2()
  hparams.ensemble_fusion_mode = "postnorm"
  return hparams

@registry.register_hparams
def transformer_base_lmlm_glue_postnorm_large_batch4():
  hparams = transformer_base_lmlm_glue_prenorm_large_batch4()
  hparams.ensemble_fusion_mode = "postnorm"
  return hparams


# DELAYED SGD

@registry.register_hparams
def transformer_base_v2_large_batch():
  hparams = transformer_base_v2()
  hparams.optimizer_multistep_accumulate_steps = 8
  hparams.optimizer = "MultistepAdam"
  return hparams


@registry.register_hparams
def transformer_base_v2_large_batch2():
  hparams = transformer_base_v2()
  hparams.optimizer_multistep_accumulate_steps = 2
  hparams.optimizer = "MultistepAdam"
  return hparams


@registry.register_hparams
def transformer_base_v2_large_batch4():
  hparams = transformer_base_v2()
  hparams.optimizer_multistep_accumulate_steps = 4
  hparams.optimizer = "MultistepAdam"
  return hparams

@registry.register_hparams
def transformer_base_v2_large_batch8():
  """Replication of Vaswani et al., 2017 on a single 12GB gpu."""
  hparams = transformer_base_v2()
  hparams.optimizer_multistep_accumulate_steps = 8
  hparams.optimizer = "MultistepAdam"
  return hparams

@registry.register_hparams
def transformer_big_large_batch2():
  hparams = transformer_big()
  #hparams.batch_size = 2048
  hparams.optimizer_multistep_accumulate_steps = 2
  hparams.optimizer = "MultistepAdam"
  return hparams

@registry.register_hparams
def transformer_big_large_batch4():
  hparams = transformer_big()
  hparams.batch_size = 2048
  hparams.optimizer_multistep_accumulate_steps = 4
  hparams.optimizer = "MultistepAdam"
  return hparams

@registry.register_hparams
def transformer_big_large_batch6():
  hparams = transformer_big_large_batch2()
  hparams.optimizer_multistep_accumulate_steps = 6
  hparams.batch_size = 3072
  return hparams

@registry.register_hparams
def transformer_big_large_batch16():
  hparams = transformer_big()
  hparams.batch_size = 2048
  hparams.optimizer_multistep_accumulate_steps = 16
  hparams.optimizer = "MultistepAdam"
  return hparams

@registry.register_hparams
def transformer_big_large_batch128():
  hparams = transformer_big()
  hparams.batch_size = 1024
  hparams.optimizer_multistep_accumulate_steps = 128
  hparams.optimizer = "MultistepAdam"
  return hparams

@registry.register_hparams
def transformer_relative_big_large_batch4():
  hparams = transformer_relative_big()
  hparams.batch_size = 2048
  hparams.optimizer_multistep_accumulate_steps = 4
  hparams.optimizer = "MultistepAdam"
  return hparams


@registry.register_hparams
def transformer_relative_big_large_batch128():
  hparams = transformer_relative_big()
  hparams.batch_size = 1024
  hparams.optimizer_multistep_accumulate_steps = 128
  hparams.optimizer = "MultistepAdam"
  return hparams


@registry.register_hparams
def transformer_base_v2_large_batch32():
  hparams = transformer_base_v2()
  hparams.optimizer_multistep_accumulate_steps = 32
  hparams.optimizer = "MultistepAdam"
  return hparams

# LABEL SMOOTHING

@registry.register_hparams
def transformer_base_v2_ls0_large_batch2():
  hparams = transformer_base_v2_large_batch2()
  hparams.label_smoothing = 0.0
  return hparams

@registry.register_hparams
def transformer_base_v2_ls02_large_batch2():
  hparams = transformer_base_v2_large_batch2()
  hparams.label_smoothing = 0.2
  return hparams

@registry.register_hparams
def transformer_base_v2_seqls_large_batch2():
  hparams = transformer_base_v2_large_batch2()
  hparams.target_modality = "symbol:seqls"
  return hparams

@registry.register_hparams
def transformer_base_v2_seqls08_large_batch2():
  hparams = transformer_base_v2_seqls_large_batch2()
  hparams.label_smoothing = 0.8
  return hparams

@registry.register_hparams
def transformer_base_v2_seqls095_large_batch2():
  hparams = transformer_base_v2_seqls_large_batch2()
  hparams.label_smoothing = 0.95
  return hparams


# LEGACY



@registry.register_hparams
def transformer_bidirectional_large_batch8():
  """Replication of Vaswani et al., 2017 on a single 12GB gpu."""
  hparams = transformer_base_v2()
  hparams.optimizer_multistep_accumulate_steps = 8
  hparams.optimizer = "MultistepAdam"
  hparams.num_decoder_layers = 4
  hparams.label_smoothing = 0.0
  hparams.add_hparam("num_bidirectional_decoder_joint_layers", 2)
  return hparams


@registry.register_hparams
def transformer_base_gibbs_large_batch8():
  """Replication of Vaswani et al., 2017 on a single 12GB gpu."""
  hparams = transformer_base_v2()
  hparams.optimizer_multistep_accumulate_steps = 8
  hparams.add_hparam("gibbs_self_attention_independence_length", 1)
  hparams.optimizer = "MultistepAdam"
  hparams.num_decoder_layers = 1
  hparams.label_smoothing = 0.0
  return hparams


@registry.register_hparams
def transformer_base_gibbs3_large_batch8():
  """Replication of Vaswani et al., 2017 on a single 12GB gpu."""
  hparams = transformer_base_gibbs_large_batch8()
  hparams.gibbs_self_attention_independence_length = 3
  return hparams


@registry.register_hparams
def transformer_base_gradout():
  """Transformer with GradOut loss.
  
  Requires the T2T fork from https://github.com/fstahlberg/tensor2tensor
  """
  hparams = transformer_base_v2()
  hparams.target_modality = "symbol:gradout"
  return hparams


@registry.register_hparams
def transformer_base_gradout_large_batch():
  """Transformer with GradOut loss.
  
  Requires the T2T fork from https://github.com/fstahlberg/tensor2tensor
  """
  hparams = transformer_base_v2()
  hparams.target_modality = "symbol:gradout"
  hparams.optimizer_multistep_accumulate_steps = 8
  hparams.optimizer = "MultistepAdam"
  return hparams


@registry.register_hparams
def transformer_base_v2_fake_replicas():
  hparams = transformer_base_v2()
  hparams.learning_rate_warmup_steps *= 8
  hparams.learning_rate /= math.sqrt(8.0)
  hparams.max_length = 150
  hparams.batch_size = 8192
  return hparams


@registry.register_hparams
def transformer_base_v2_small_lr():
  hparams = transformer_base_v2()
  hparams.learning_rate /= math.sqrt(8.0)
  hparams.max_length = 150
  hparams.batch_size = 8192
  return hparams

@registry.register_hparams
def slicenet_large_batch2():
  hparams = slicenet_params1_noam()
  hparams.optimizer_multistep_accumulate_steps = 2
  hparams.optimizer = "MultistepAdam"
  hparams.batch_size = 4096
  hparams.max_length = 150
  return hparams
