'''
 Copyright (c) 2023-2026 David Lanzendörfer <leviathan@libresilicon.com>
 Distributed under the GNU GPL v2. For full terms see the file docs/COPYING.
'''

from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from misesgpt.html_fetcher import MisesHTMLBookCatalog
from misesgpt.epub_fetcher import MisesEPUBookCatalog

from torch.utils.data import Dataset

class MisesDataset(Dataset):
  def __init__(self, tokenizer, max_length):
    assert(tokenizer is not None)

    self.tokenizer = tokenizer
    self.sentences = []
    self.max_length = max_length

    self.html_books = MisesHTMLBookCatalog()
    self.epub_books = MisesEPUBookCatalog()
    self.get_paragraphs()

  def process_paragraph(self, p):
    tokens = self.tokenizer.encode('<|startoftext|>' + p + '<|endoftext|>', add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True, padding=True)
    if len(tokens) <= self.max_length:
      s = '<|startoftext|>' + p + '<|endoftext|>' + (self.max_length-len(pen.input_ids))*'<|pad|>'
      pen = self.tokenizer(s, add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True, padding=True)
      self.sentences.append(pen)
    return True

  def get_paragraphs(self):
    print("Reading all the JSON files")
    paragraphs = []
    for book in self.html_books.books_json:
      paragraphs += self.html_books.books_json[book]
    for book in self.epub_books.books_json:
      paragraphs += self.epub_books.books_json[book]

    print("Filling samples array")
    with tqdm(total=len(paragraphs)) as pbar:
      with ThreadPoolExecutor() as ex:
        futures = [ex.submit(self.process_paragraph, p) for p in paragraphs]
        for future in as_completed(futures):
          result = future.result()
          pbar.update(1)

  def __getitem__(self, item):
    return self.sentences[item]

  def __len__(self):
    return len(self.sentences)
