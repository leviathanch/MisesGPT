'''
 Copyright (c) 2023-2026 David Lanzendörfer <leviathan@libresilicon.com>
 Distributed under the GNU GPL v2. For full terms see the file docs/COPYING.
'''

from transformers import (
	ReformerForMaskedLM,
    ReformerModelWithLMHead,
    ReformerTokenizerFast,
  )

tokenizer = ReformerTokenizerFast.from_pretrained('remitokenizer')
model = ReformerModelWithLMHead.from_pretrained('model')
model.eval()

test_strings = [
    "<s>The static state can",
]

for text in test_strings:
  x = tokenizer(text, return_tensors = 'pt', padding = True, add_special_tokens = True)
  y = model.generate(**x, do_sample=True, temperature=0.5, num_beams=5, max_length=20)
  y = tokenizer.batch_decode(y, skip_special_tokens = True)
  for yi in y:
      print('\nMisesGPT:',yi)
