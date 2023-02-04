'''
 Copyright (c) 2023-2026 David Lanzendörfer <leviathan@libresilicon.com>
 Distributed under the GNU GPL v2. For full terms see the file docs/COPYING.
'''
import pickle
import hashlib

from os.path import exists, join
from os import mkdir

from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from misesgpt.html_fetcher import MisesHTMLBookCatalog
from misesgpt.epub_fetcher import MisesEPUBookCatalog
from misesgpt.basic_words import MisesBasicWords

from torch.utils.data import Dataset

class MisesDataset(Dataset):
  cache_folder = 'mises_dataset_cache'
  book_sentences = None

  def __init__(self, tokenizer, max_length, only_build_cache=False):
    assert(tokenizer is not None)
    self.only_build_cache = only_build_cache
    self.tokenizer = tokenizer
    self.sentences = []
    self.max_length = max_length

    if not exists(self.cache_folder):
      mkdir(self.cache_folder)

    print("Reading basic vocabulary...")
    word_definitions = MisesBasicWords()
    self.get_basic_words(word_definitions.words)
    print("Reading HTML books...")
    html_books = MisesHTMLBookCatalog()
    self.get_paragraphs(html_books.books_json)
    print("Reading EPUBS...")
    epub_books = MisesEPUBookCatalog()
    self.get_paragraphs(epub_books.books_json)

  def process_paragraph(self, p):
    tokens = self.tokenizer.encode('<s>' + p + '</s>', add_special_tokens=True, padding=True)
    if len(tokens) <= self.max_length:
      s = '<s>' + p + '</s>' + (self.max_length-len(tokens))*'<pad>'
      pen = self.tokenizer(s, add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True, padding=True)
      self.book_sentences.append(pen)
    return True

  def get_basic_words(self, words):
    pickle_file = join(self.cache_folder,'words.pickle')
    if exists(pickle_file):
      if not self.only_build_cache:
        with open(pickle_file,'rb') as f:
          self.sentences += pickle.load(f)
          f.close()
      return True

    for i, word in enumerate(words):
      print("Processing word", word, '(',i,'of',len(words),')')
      self.book_sentences = []
      with tqdm(total=len(words[word])) as pbar:
        with ThreadPoolExecutor() as ex:
          futures = [ex.submit(self.process_paragraph, p) for p in words[word]]
          for future in as_completed(futures):
            result = future.result()
            pbar.update(1)
      if not self.only_build_cache:
        self.sentences += self.book_sentences

    with open(pickle_file,'wb') as f:
      pickle.dump( self.book_sentences, f)
      f.close()

  def get_paragraphs(self, books_json):
    paragraphs = []
    for i, book in enumerate(books_json):
      pickle_file = join(self.cache_folder,hashlib.sha256(book.encode()).hexdigest()+'.pickle')
      if exists(pickle_file):
        if not self.only_build_cache:
          with open(pickle_file,'rb') as f:
            self.sentences += pickle.load(f)
            f.close()
        continue

      print("Processing entry", book, '(',i,'of',len(books_json),')')
      self.book_sentences = []
      with tqdm(total=len(books_json[book])) as pbar:
        with ThreadPoolExecutor() as ex:
          futures = [ex.submit(self.process_paragraph, p) for p in books_json[book]]
          for future in as_completed(futures):
            result = future.result()
            pbar.update(1)

      if not self.only_build_cache:
        self.sentences += self.book_sentences
      with open(pickle_file,'wb') as f:
        pickle.dump( self.book_sentences, f)
        f.close()

  def __getitem__(self, item):
    return self.sentences[item]

  def __len__(self):
    return len(self.sentences)
