'''
 Copyright (c) 2023-2026 David Lanzendörfer <leviathan@libresilicon.com>
 Distributed under the GNU GPL v2. For full terms see the file docs/COPYING.
'''

from transformers import (
    ReformerTokenizerFast,
  )

from misesgpt.dataset import MisesDataset

tokenizer = ReformerTokenizerFast(
  vocab_file = 'model/remilamda.model',
  return_special_tokens_mask = True,
  bos_token = '<s>',
  eos_token = '</s>',
  pad_token = '<pad>',
  unk_token = '<unk>',
  mask_token = '<mask>',
)
dataset = MisesDataset(tokenizer, sequence_length=64*64, only_build_cache=True)
