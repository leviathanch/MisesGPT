'''
 Copyright (c) 2023-2026 David Lanzendörfer <leviathan@libresilicon.com>
 Distributed under the GNU GPL v2. For full terms see the file docs/COPYING.
'''

from transformers import (
    ReformerTokenizerFast,
  )

from misesgpt.dataset import MisesDataset

max_length =  64*64 # "axial_pos_shape": [64, 64]

tokenizer = ReformerTokenizerFast.from_pretrained(
  'model',
  return_special_tokens_mask = True,
  bos_token = '<s>',
  eos_token = '</s>',
  pad_token = '<pad>',
  unk_token = '<unk>',
  mask_token = '<mask>',
)
dataset = MisesDataset(tokenizer, max_length, only_build_cache=True)
