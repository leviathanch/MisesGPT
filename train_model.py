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
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
  )

from torch.utils.data import Dataset, random_split
from misesgpt.html_fetcher import MisesHTMLBookCatalog
from misesgpt.epub_fetcher import MisesEPUBookCatalog

print("Preparing tokenizer")

dscache = './dataset_cache/'
tokenizer = GPT2Tokenizer.from_pretrained(
  'gpt2',
  bos_token = '<|startoftext|>',
  eos_token = '<|endoftext|>',
  pad_token = '<|pad|>'
)

print("Preparing training data")

def append_data(file, books):
  maxlen = 0
  for book in books:
    for p in books[book]:
      s = '<|startoftext|>' + p + '<|endoftext|>'
      sl = len(tokenizer.encode(s))
      if sl < 1024:
        maxlen = sl if sl > maxlen else maxlen
        file.write(s+'\n')
  return maxlen

# prepare training data:
df = 'data.txt'
lens = []
if exists(df):
  remove(df)
with open(df,'a') as file:
  lens.append(append_data(file, MisesHTMLBookCatalog().books_json))
  lens.append(append_data(file, MisesEPUBookCatalog().books_json))
  file.close()

max_length = max(lens)
assert(max_length > 0)
block_size = pow(2, ceil(log(max_length)/log(2)));
print("Maxlen is", max_length)
print("Block size is", block_size)

# ---------
print("Reinitialize tokenizer")
tokenizer = GPT2Tokenizer.from_pretrained(
  'gpt2',
  bos_token = '<|startoftext|>',
  eos_token = '<|endoftext|>',
  pad_token = '<|pad|>',
  max_length = max_length,
)
# ---------

print("Preparing data set")

if not exists(dscache):
  mkdir(dscache)

dataset = TextDataset(
  tokenizer = tokenizer,
  file_path = './data.txt',
  block_size = block_size,
  cache_dir = dscache,
)
train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

print("Create/load model")

config = GPT2Config(n_layer=6, n_head=6, n_embd=192)
if exists('./base_model'):
  GPT2LMHeadModel.from_pretrained('./base_model')
else:
  model = GPT2LMHeadModel(config)
  model.resize_token_embeddings(len(tokenizer))

training_args =TrainingArguments(
  output_dir = './model',
  num_train_epochs = 10,
  warmup_steps = 500,
  save_steps = 500,
)

print("Set up collator")

data_collator = DataCollatorForLanguageModeling(
  tokenizer = tokenizer,
  mlm = False,
)

print("Run training")

trainer = Trainer(
  model = model,
  args = training_args,
  data_collator = data_collator,
  train_dataset = train_dataset,
  eval_dataset = val_dataset,
)

trainer.train()
trainer.save_model()
