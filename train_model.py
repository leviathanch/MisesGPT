'''
 Copyright (c) 2023-2026 David Lanzendörfer <leviathan@libresilicon.com>
 Distributed under the GNU GPL v2. For full terms see the file docs/COPYING.
'''

from os.path import exists
from os import mkdir
from os import remove

from math import ceil, log

from transformers import (
    ReformerConfig,
    ReformerModelWithLMHead,
    ReformerTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
  )

from torch.utils.data import random_split
from misesgpt.dataset import MisesDataset

max_length = 4096

tokenizer = ReformerTokenizer.from_pretrained(
  'model',
  return_special_tokens_mask = True,
  bos_token = '<|startoftext|>',
  eos_token = '<|endoftext|>',
  pad_token = '<|pad|>',
  unk_token = '<|unk|>',
  mask_token = '<|mask|>',
)
dataset = MisesDataset(tokenizer, max_length)

train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

config = ReformerConfig(is_decoder=True, num_buckets=128)
if exists('./model/config.json'):
  print("Found existing model. Loading...")
  model = ReformerModelWithLMHead.from_pretrained('./model')
  model.eval()
else:
  model = ReformerModelWithLMHead(config)
  model.resize_token_embeddings(len(tokenizer))

training_args = TrainingArguments(
  output_dir = './model',
  num_train_epochs = 10,
  warmup_steps = 500,
  save_steps = 10000,
)

data_collator = DataCollatorForLanguageModeling(
  tokenizer = tokenizer,
  mlm = True,
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
