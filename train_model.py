'''
 Copyright (c) 2023-2026 David Lanzendörfer <leviathan@libresilicon.com>
 Distributed under the GNU GPL v2. For full terms see the file docs/COPYING.
'''
from transformers import utils
utils.logging.set_verbosity_error()  # Suppress standard warnings

import bitsandbytes as bnb
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from os.path import exists
from os import mkdir
from os import remove
from os.path import join
from math import ceil, log
import argparse
import datetime

from transformers import (
  ReformerConfig,
  ReformerModelWithLMHead,
  ReformerTokenizerFast,
  DataCollatorForLanguageModeling,
  Trainer,
  TrainingArguments,
)

from torch import nn
from transformers.trainer_pt_utils import get_parameter_names

from accelerate import Accelerator

from torch.utils.data import random_split
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.cuda import set_device
import tensorflow as tf

from misesgpt.dataset import MisesDataset

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'runs/' + current_time
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

max_length = 64*64

tokenizer = ReformerTokenizerFast(
  vocab_file = 'model/remilamda.model',
  return_special_tokens_mask = True,
  add_special_tokens=True,
  padding = True,
  bos_token = '<s>',
  eos_token = '</s>',
  pad_token = '<pad>',
  unk_token = '<unk>',
  mask_token = '<mask>',
)

def get_optimizer(model, args):
  default_args = {
      "output_dir": "tmp",
      "evaluation_strategy": "steps",
      "num_train_epochs": 1,
      "log_level": "error",
      "report_to": "none",
  }
  training_args = TrainingArguments(per_device_train_batch_size=args.batch_size, **default_args)
  decay_parameters = get_parameter_names(model, [nn.LayerNorm])
  decay_parameters = [name for name in decay_parameters if "bias" not in name]
  optimizer_grouped_parameters = [
      {
          "params": [p for n, p in model.named_parameters() if n in decay_parameters],
          "weight_decay": training_args.weight_decay,
      },
      {
          "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
          "weight_decay": 0.0,
      },
  ]

  optimizer_kwargs = {
      "betas": (training_args.adam_beta1, training_args.adam_beta2),
      "eps": training_args.adam_epsilon,
  }
  optimizer_kwargs["lr"] = training_args.learning_rate
  adam_bnb_optim = bnb.optim.Adam8bit(
      optimizer_grouped_parameters,
      betas=(training_args.adam_beta1, training_args.adam_beta2),
      eps=training_args.adam_epsilon,
      lr=training_args.learning_rate,
  )
  return adam_bnb_optim

def get_data(args):
  dataset = MisesDataset(tokenizer, max_length, cached_only=True)
  data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer,
    mlm = True,
  )
  loader = DataLoader(dataset, collate_fn=data_collator, batch_size=args.batch_size)
  return loader

def get_model():
  if exists('./model/config.json'):
    print("Found existing model. Loading...")
    model = ReformerModelWithLMHead.from_pretrained('./model')
    model.eval()
    warmup = 0
  else:
    cfg = {
      "attention_head_size": 64,
      "attention_probs_dropout_prob": 0.1,
      "attn_layers": [
        "local",
        "lsh",
        "local",
        "lsh",
        "local",
        "lsh"
      ],
    "sinusoidal_pos_embds": True,
      "axial_norm_std": 1.0,
      "axial_pos_embds": True,
      "axial_pos_embds_dim": [32, 96],
      "axial_pos_shape": [64, 64],
      "chunk_size_feed_forward": 0,
      "chunk_size_lm_head": 0,
      "eos_token_id": 2,
      "feed_forward_size": 512,
      "hidden_act": "relu",
      "hidden_dropout_prob": 0.05,
      "hidden_size": 128,
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "is_decoder": True,
      "layer_norm_eps": 1e-12,
      "local_attn_chunk_length": 64,
      "lsh_attn_chunk_length": 64,
      "max_position_embeddings": 524288,
      "model_type": "reformer",
      "num_attention_heads": 2,
      "num_buckets": [64, 128],
      "num_chunks_after": 0,
      "num_chunks_before": 1,
      "num_hashes": 1,
      "num_hidden_layers": 6,
      "output_past": True,
      "pad_token_id": 0,
      "task_specific_params": {
        "text-generation": {
          "do_sample": True,
          "max_length": 100
        }
      },
      "vocab_size": 52000
    }
    config = ReformerConfig(**cfg)
    model = ReformerModelWithLMHead(config)
    model.resize_token_embeddings(len(tokenizer))
  return model

def train_cuda(gpu, args):
  train_loader = args.train_loader
  r = args.nr*args.cores+gpu
  dist.init_process_group(
    'nccl',
    rank=r,
    world_size=args.world_size
  )
  set_device(gpu)
  model = DistributedDataParallel(args.model.cuda(), device_ids=[gpu])
  model = model.train().to('cuda')

  optimizer = get_optimizer(model, args)
  total_step = len(train_loader)
  
  for epoch in range(args.epochs):
    for i, item in enumerate(train_loader):
      x = item.to('cuda')
      loss = model(**x).loss
      # Backward and optimize
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if (i + 1) % 100 == 0 and gpu == 0:
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
          epoch + 1, 
          args.epochs,
          i + 1, 
          total_step,
          loss.item())
        )
        with train_summary_writer.as_default():
          tf.summary.scalar('loss', loss.item(), step=i+1)

      if (i + 1) % args.save_steps == 0 and gpu == 0:
        output_dir = join(args.output_dir, 'checkpoint-{}'.format(epoch*total_step+i+1))
        print('Savin checkpoint',output_dir)
        model.module.save_pretrained(output_dir)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--use-cuda', default=True, type=bool, help='Whether to use CUDA')
  parser.add_argument('--use-tpus', default=True, type=bool, help='Whether to use TPUs')
  parser.add_argument('-bs', '--batch-size', default=1, type=int, help='size of a batch')
  parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
  parser.add_argument('-c', '--cores', default=1, type=int, help='number of cores per node')
  parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
  parser.add_argument('--output-dir', default='./out', type=str, help='Output directory')
  parser.add_argument('--epochs', default=2, type=int, metavar='N', help='number of total epochs to run')
  parser.add_argument('--save-steps', default=100, type=int, help='Save every N steps')
  args = parser.parse_args()
  args.world_size = args.cores*args.nodes
  args.train_loader = get_data(args)
  args.model = get_model()

  if args.use_cuda:
    mp.spawn(train_cuda, nprocs=args.cores, args=(args,), join=True)
  elif args.use_tpus: # TODO: support TPUs
    pass

if __name__ == '__main__':
  main()
