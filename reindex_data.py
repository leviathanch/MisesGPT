'''
 Copyright (c) 2023-2026 David Lanzendörfer <leviathan@libresilicon.com>
 Distributed under the GNU GPL v2. For full terms see the file docs/COPYING.
'''

from transformers import (
    ReformerTokenizer,
  )

from misesgpt.dataset import MisesDataset

max_length = 4096

tokenizer = ReformerTokenizer.from_pretrained(
  'model',
  return_special_tokens_mask = True,
  bos_token = '<s>',
  eos_token = '</s>',
  pad_token = '<pad>',
  unk_token = '<unk>',
  mask_token = '<mask>',
)
dataset = MisesDataset(tokenizer, max_length)
