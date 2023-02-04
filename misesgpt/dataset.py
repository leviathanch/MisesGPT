'''
 Copyright (c) 2023-2026 David Lanzendörfer <leviathan@libresilicon.com>
 Distributed under the GNU GPL v2. For full terms see the file docs/COPYING.
'''
import pickle

from os.path import exists

from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from misesgpt.html_fetcher import MisesHTMLBookCatalog
from misesgpt.epub_fetcher import MisesEPUBookCatalog

from torch.utils.data import Dataset

class MisesDataset(Dataset):
  cache_file = 'mises_dataset_cache.pickle'

  def __init__(self, tokenizer, max_length):
    assert(tokenizer is not None)
    self.tokenizer = tokenizer
    self.sentences = []
    self.max_length = max_length

    if exists(self.cache_file):
      print("Loading cached dataset",self.cache_file)
      with open(self.cache_file,'rb') as f:
        self.sentences = pickle.load(f)
        f.close()
    else:
      print("Reading HTML books...")
      html_books = MisesHTMLBookCatalog()
      self.get_paragraphs(html_books.books_json)
      print("Reading EPUBS...")
      epub_books = MisesEPUBookCatalog()
      self.get_paragraphs(epub_books.books_json)
      print("Writing dataset cache",self.cache_file)
      with open(self.cache_file,'wb') as f:
        pickle.dump( self.sentences, f)
        f.close()

  def process_paragraph(self, p):
    tokens = self.tokenizer.encode('<|startoftext|>' + p + '<|endoftext|>', add_special_tokens=True, padding=True)
    if len(tokens) <= self.max_length:
      s = '<|startoftext|>' + p + '<|endoftext|>' + (self.max_length-len(tokens))*'<|pad|>'
      pen = self.tokenizer(s, add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True, padding=True)
      self.sentences.append(pen)
    return True

  def get_paragraphs(self, books_json):
    paragraphs = []
    for i, book in enumerate(books_json):
      print("Processing book",i)
      with tqdm(total=len(books_json[book])) as pbar:
        with ThreadPoolExecutor() as ex:
          futures = [ex.submit(self.process_paragraph, p) for p in books_json[book]]
          for future in as_completed(futures):
            result = future.result()
            pbar.update(1)

  def __getitem__(self, item):
    return self.sentences[item]

  def __len__(self):
    return len(self.sentences)
