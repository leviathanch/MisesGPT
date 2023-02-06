'''
 Copyright (c) 2023-2026 David Lanzendörfer <leviathan@libresilicon.com>
 Distributed under the GNU GPL v2. For full terms see the file docs/COPYING.
'''

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from os.path import exists
from os import mkdir
from os import remove
from math import ceil, log

from transformers import (
    ReformerConfig,
	ReformerModelWithLMHead,
    ReformerTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    utils,
  )

from torch.utils.data import random_split
from torch.nn import DataParallel
from misesgpt.dataset import MisesDataset

utils.logging.set_verbosity_error()  # Suppress standard warnings

max_length = 64*64

tokenizer = ReformerTokenizerFast.from_pretrained(
  'model',
  return_special_tokens_mask = True,
  bos_token = '<s>',
  eos_token = '</s>',
  pad_token = '<pad>',
  unk_token = '<unk>',
  mask_token = '<mask>',
)
dataset = MisesDataset(tokenizer, max_length, cached_only=True)

train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

if exists('./model/config.json'):
  print("Found existing model. Loading...")
  model = ReformerModelWithLMHead.from_pretrained('./model')
  model.eval()
  warmup = 0
else:
  cfg = {
    "attention_head_size": 64,
    "attn_layers": ["local", "lsh", "local", "lsh", "local", "lsh"],
    "axial_pos_embds": True,
    "sinusoidal_pos_embds": False,
    #"axial_pos_embds_dim": [64, 192],
    #"axial_pos_shape": [512, 1024],
    "axial_pos_shape": [64, 64],
    "lsh_attn_chunk_length": 64,
    "local_attn_chunk_length": 64,
    "feed_forward_size": 512,
    "hidden_act": "relu",
    "hidden_size": 256,
    "is_decoder": True,
    "max_position_embeddings": 524288,
    "num_attention_heads": 2,
    "num_buckets": [64, 128],
    "num_hashes": 1,
    "vocab_size": 320,
    "lsh_attention_probs_dropout_prob": 0.0,
    "lsh_num_chunks_before": 1,
    "lsh_num_chunks_after": 0,
    "local_num_chunks_before": 1,
    "local_num_chunks_after": 0,
    "local_attention_probs_dropout_prob": 0.025,
    "hidden_dropout_prob": 0.025,
  }
  config = ReformerConfig(**cfg)
  model = ReformerModelWithLMHead(config)
  model.resize_token_embeddings(len(tokenizer))
  model.train()
  warmup = 500


training_args = {
    "learning_rate": 1e-3,
    "max_steps": 2000,
    "do_train": True,
    "gradient_accumulation_steps": 8,
    "logging_steps": 50,
    "warmup_steps": warmup,
    "weight_decay": 0.001,
    "per_gpu_train_batch_size": 1,
    "per_gpu_eval_batch_size": 1,
    "save_steps": 50,
    "output_dir": "./model"
}
training_args = TrainingArguments(**training_args)

data_collator = DataCollatorForLanguageModeling(
  tokenizer = tokenizer,
  mlm = True,
)

net = DataParallel(model)
trainer = Trainer(
  model = net,
  args = training_args,
  data_collator = data_collator,
  train_dataset = train_dataset,
  eval_dataset = val_dataset,
)

trainer.train()
trainer.save_model()
