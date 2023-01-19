'''
 Copyright (c) 2023-2026 David Lanzendörfer <leviathan@libresilicon.com>
 Distributed under the GNU GPL v2. For full terms see the file docs/COPYING.
'''

from os.path import exists
from os import mkdir
from os import remove

from math import ceil, log

from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
  )

from torch.utils.data import random_split
from misesgpt.dataset import MisesDataset

max_length = 1024

tokenizer = GPT2Tokenizer.from_pretrained(
  'gpt2',
  return_special_tokens_mask = True,
  bos_token = '<|startoftext|>',
  eos_token = '<|endoftext|>',
  pad_token = '<|pad|>'
)
dataset = MisesDataset(tokenizer, max_length)

train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

config = GPT2Config(n_layer=6, n_head=6, n_embd=192)
if exists('./base_model'):
  model = GPT2LMHeadModel.from_pretrained('./base_model')
  model.eval()
else:
  model = GPT2LMHeadModel(config)
  model.resize_token_embeddings(len(tokenizer))

training_args =TrainingArguments(
  output_dir = './model',
  num_train_epochs = 10,
  warmup_steps = 500,
  save_steps = 10000,
)

data_collator = DataCollatorForLanguageModeling(
  tokenizer = tokenizer,
  mlm = False,
)

trainer = Trainer(
  model = model,
  args = training_args,
  data_collator = data_collator,
  train_dataset = train_dataset,
  eval_dataset = val_dataset,
)

trainer.train()
trainer.save_model()
