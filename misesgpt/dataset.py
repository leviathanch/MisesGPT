'''
 Copyright (c) 2023-2026 David Lanzendörfer <leviathan@libresilicon.com>
 Distributed under the GNU GPL v2. For full terms see the file docs/COPYING.
'''
import pickle
import hashlib

from os.path import exists, join
from os import mkdir

from glob import glob

from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from misesgpt.html_fetcher import MisesHTMLBookCatalog
from misesgpt.epub_fetcher import MisesEPUBookCatalog
from misesgpt.basic_words import MisesBasicWords

from torch.utils.data import Dataset

class MisesDataset(Dataset):
  cache_folder = 'mises_dataset_cache'
  book_sentences = None
  book_items = None

  def __init__(self, tokenizer, max_length, only_build_cache=False, cached_only=False):
    assert(tokenizer is not None)
    self.only_build_cache = only_build_cache
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.items = []
    
    if not exists(self.cache_folder):
      mkdir(self.cache_folder)

    if cached_only:
      print("Only loading already cached pickles")
      pickles = glob(join(self.cache_folder,'*.pickle'))
      with tqdm(total=len(pickles)) as pbar:
        with ThreadPoolExecutor() as ex:
          futures = [ex.submit(self.pickle_loading_threat, p) for p in pickles]
          for future in as_completed(futures):
            result = future.result()
            pbar.update(1)
    else:
      print("Reading basic vocabulary...")
      word_definitions = MisesBasicWords()
      self.get_basic_words(word_definitions.words)
      print("Reading HTML books...")
      html_books = MisesHTMLBookCatalog()
      self.get_paragraphs(html_books.books_json)
      print("Reading EPUBS...")
      epub_books = MisesEPUBookCatalog()
      self.get_paragraphs(epub_books.books_json)

  def pickle_loading_threat(self, pickle_file):
    with open(pickle_file,'rb') as f:
      self.items += pickle.load(f)
      f.close()

  def process_paragraph(self, p):
    s = '<s>' + p + '</s>\n'
    new_tokens = self.tokenizer.encode(s, add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)
    if len(new_tokens)+self.book_fragments['length'] <= self.max_length:
      self.book_fragments['text'] += s
      self.book_fragments['length'] += len(new_tokens)
    else:
      text_tokens = self.tokenizer.encode(self.book_fragments['text'], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)
      self.book_fragments['text'] += '<pad>'*(self.max_length-len(text_tokens))
      pen = self.tokenizer(self.book_fragments['text'], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)
      self.book_items.append(pen)

    return True

  def get_basic_words(self, word_dict):
    pickle_file = join(self.cache_folder,'words.pickle')
    if exists(pickle_file):
      if not self.only_build_cache:
        with open(pickle_file,'rb') as f:
          self.items += pickle.load(f)
          f.close()
      return True
    words = []
    for word in word_dict:
      words += word_dict[word]

    print("Processing words...")
    self.book_fragments = { 'length':0, 'text':""}
    self.book_items = []
    with tqdm(total=len(words)) as pbar:
      with ThreadPoolExecutor() as ex:
        futures = [ex.submit(self.process_paragraph, p) for p in words]
        for future in as_completed(futures):
          result = future.result()
          pbar.update(1)

    if not self.only_build_cache:
      self.items += self.book_items

    with open(pickle_file,'wb') as f:
      pickle.dump( self.book_items, f)
      f.close()

  def get_paragraphs(self, books_json):
    paragraphs = []
    for i, book in enumerate(books_json):
      pickle_file = join(self.cache_folder,hashlib.sha256(book.encode()).hexdigest()+'.pickle')
      if exists(pickle_file):
        if not self.only_build_cache:
          with open(pickle_file,'rb') as f:
            self.items += pickle.load(f)
            f.close()
        continue

      print("Processing entry", book, '(',i,'of',len(books_json),')')
      self.book_fragments = { 'length':0, 'text':""}
      self.book_items = []
      with tqdm(total=len(books_json[book])) as pbar:
        with ThreadPoolExecutor() as ex:
          futures = [ex.submit(self.process_paragraph, p) for p in books_json[book]]
          for future in as_completed(futures):
            result = future.result()
            pbar.update(1)

      if not self.only_build_cache:
        self.items += self.book_items
      with open(pickle_file,'wb') as f:
        pickle.dump( self.book_items, f)
        f.close()

  def __getitem__(self, item):
    return self.items[item]

  def __len__(self):
    return len(self.items)
