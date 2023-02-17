'''
 Copyright (c) 2023-2026 David Lanzendörfer <leviathan@libresilicon.com>
 Distributed under the GNU GPL v2. For full terms see the file docs/COPYING.
'''

from transformers import (
    ReformerTokenizerFast,
  )

from misesgpt.dataset import MisesDataset

tokenizer = ReformerTokenizerFast.from_pretrained('remitokenizer')
dataset = MisesDataset(tokenizer, sequence_length=2048, chunk_length=64, only_build_cache=True)
