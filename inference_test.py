'''
 Copyright (c) 2023-2026 David Lanzendörfer <leviathan@libresilicon.com>
 Distributed under the GNU GPL v2. For full terms see the file docs/COPYING.
'''

from transformers import (
	ReformerForMaskedLM,
    ReformerModelWithLMHead,
    ReformerTokenizerFast,
  )

tokenizer = ReformerTokenizerFast.from_pretrained(
  'model',
  return_special_tokens_mask = True,
)

model = ReformerModelWithLMHead.from_pretrained('model')
model.eval()

test_strings = [
    "No less has this been true of economics.",
]

for text in test_strings:
  x = tokenizer(text, return_tensors = 'pt', padding=True )
  y = model.generate(**x, do_sample=True)# max_length=1024)
  y = tokenizer.batch_decode(y, skip_special_tokens = True )
  for yi in y:
      print('\nMisesGPT:',yi)
