'''
 Copyright (c) 2023-2026 David Lanzendörfer <leviathan@libresilicon.com>
 Distributed under the GNU GPL v2. For full terms see the file docs/COPYING.
'''

from transformers import (
    GPT2Model,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
  )

tokenizer = GPT2Tokenizer.from_pretrained(
  'gpt2',
#  bos_token = '<|startoftext|>',
#  eos_token = '<|endoftext|>',
#  pad_token = '<|pad|>',
)

model = GPT2Model.from_pretrained('./base_model')
model.eval()

test_strings = [
  "Taxation is ",
  "War is ",
]

for text in test_strings:
  #print("Encode\n")
  x = tokenizer.encode(text, return_tensors = 'pt')
  #print("Generate\n")
  y = model.generate(x, max_length=50, pad_token = '<|pad|>')
  #print("Decode\n")
  for beam in y:
    beamd = tokenizer.decode(beam, skip_special_tokens = True, max_new_tokens = 50)
    print(beamd)
