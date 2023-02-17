'''
 Copyright (c) 2023-2026 David Lanzendörfer <leviathan@libresilicon.com>
 Distributed under the GNU GPL v2. For full terms see the file docs/COPYING.
'''
from transformers import utils
utils.logging.set_verbosity_error()  # Suppress standard warnings

from os.path import exists
from os import mkdir
from os import remove
from os.path import join
import os
from math import ceil, log
import argparse
import datetime
import json

from transformers import (
  ReformerConfig,
  ReformerModelWithLMHead,
  ReformerTokenizerFast,
  DataCollatorForLanguageModeling,
  Trainer,
  TrainingArguments,
  set_seed
)
from transformers.optimization import get_linear_schedule_with_warmup

from torch import nn
from transformers.trainer_pt_utils import get_parameter_names

from torch.utils.data import random_split
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.cuda import set_device
import tensorflow as tf

from misesgpt.dataset import MisesDataset

max_length = 32*64

def get_tokenizer():
  tokenizer = ReformerTokenizerFast.from_pretrained(
    'remitokenizer',
    return_special_tokens_mask = True,
    add_special_tokens = True,
    max_length = max_length,
    padding = 'max_length',
  )
  return tokenizer

def get_model(tokenizer):
  if exists('./model/config.json'):
    print("Found model config...")
    if exists('./model/pytorch_model.bin'):
        print("Found pretrained model...")
        model = ReformerModelWithLMHead.from_pretrained('./model')
        warmup = 0
    else:
      with open('./model/config.json', 'r') as f:
        cfg = json.load(f)
        f.close()
      config = ReformerConfig(**cfg)
      model = ReformerModelWithLMHead(config)
      model.resize_token_embeddings(len(tokenizer))
  else:
    raise Exception("No model found")

  return model

def train_cuda(args):
  os.environ["WANDB_DISABLED"] = "true"
  os.environ["WORLD_SIZE"] = str(args.world_size)
  os.environ["MASTER_ADDR"] = str(args.master_address)
  os.environ["MASTER_PORT"] = str(args.master_port)
  gpu = args.use_core

  training_args = {
      "max_steps": args.max_steps,
      "do_train": True,
      "evaluation_strategy": 'steps',
      "logging_steps": args.log_steps,
      "warmup_steps": args.warmup_steps,
      "per_device_train_batch_size": args.batch_size,
      "save_steps": args.save_steps,
      "output_dir": args.output_dir,
  }
  if args.weight_decay is not None:
    training_args["weight_decay"] = args.weight_decay
  if args.learning_rate is not None:
    training_args["learning_rate"] = args.learning_rate
  if args.gradient_accumulation_steps is not None:
    training_args["gradient_accumulation_steps"] = args.gradient_accumulation_steps

  if args.distributed:
    set_device(gpu)
    training_args["local_rank"] = args.rank*args.cores+gpu
    os.environ["RANK"] = str(args.rank*args.cores+gpu)

  train_size = int(0.9 * len(args.dataset))
  train_dataset, val_dataset = random_split(args.dataset, [train_size, len(args.dataset) - train_size])

  training_args = TrainingArguments(**training_args)
  trainer = Trainer(
    model = DistributedDataParallel(args.model.cuda(), device_ids=[gpu]) if args.distributed else args.model,
    args = training_args,
    data_collator = args.data_collator,
	train_dataset = args.dataset,
    eval_dataset = val_dataset,
  )
  trainer.train()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--distributed', action='store_true', help='Turn on distributed mode')
  parser.add_argument('-bs', '--batch-size', default=1, type=int, help='size of a batch')
  parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
  parser.add_argument('-c', '--cores', default=1, type=int, help='number of cores per node')
  parser.add_argument('-r', '--ranking', default=0, type=int, help='ranking within the nodes')
  parser.add_argument('--output-dir', default='./out', type=str, help='Output directory')
  parser.add_argument('--max-steps', default=2000, type=int, metavar='N', help='Maximums steps')
  parser.add_argument('--save-steps', default=100, type=int, help='Save every N steps')
  parser.add_argument('--warmup-steps', default=50, type=int, help='Warmup steps')
  parser.add_argument('--log-steps', default=100, type=int, help='Log every N steps')
  parser.add_argument('--gradient-accumulation-steps', type=int, help='Gradient accumulation steps')
  parser.add_argument('--use-core', default=0, type=int, help='The core to spin up on')
  parser.add_argument('--master-address', default="127.0.0.1", type=str, help='The master address')
  parser.add_argument('--master-port', default="8888", type=str, help='The master port')
  parser.add_argument('--learning-rate', default=1e-3, type=float, help='Learning rate')
  parser.add_argument('--weight-decay', type=float, help='Weight decay')
  parser.add_argument('--seed', type=int, help='Seed')

  args = parser.parse_args()
  args.world_size = args.cores*args.nodes
  tokenizer = get_tokenizer()
  if args.seed is not None:
    set_seed(args.seed)
  args.model = get_model(tokenizer)
  args.dataset = MisesDataset(tokenizer, max_length, 64, cached_only=True)
  args.data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer,
    #mlm = False,
    mlm = True,
    mlm_probability = 0.15,
    return_tensors = 'pt',
  )
  train_cuda(args)

if __name__ == '__main__':
  main()
