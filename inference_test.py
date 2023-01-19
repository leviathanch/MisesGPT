'''
 Copyright (c) 2023-2026 David Lanzendörfer <leviathan@libresilicon.com>
 Distributed under the GNU GPL v2. For full terms see the file docs/COPYING.
'''

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
  )

tokenizer = GPT2Tokenizer.from_pretrained(
  'gpt2',
  return_special_tokens_mask = True,
  bos_token = '<|startoftext|>',
  eos_token = '<|endoftext|>',
  pad_token = '<|pad|>'
)

model = GPT2LMHeadModel.from_pretrained('./base_model')
model.eval()

test_strings = [
    "Taxation is",
    "War is",
]

for text in test_strings:
  x = tokenizer(text, return_tensors = 'pt', padding=True )
  y = model.generate(**x, do_sample=True, temperature=0.9, max_length=150)
  y = tokenizer.decode(y[0], skip_special_tokens = True)
  print('\n\n'+y+'\n\n')
