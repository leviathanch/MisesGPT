import multiprocessing

from misesgpt.html_fetcher import MisesHTMLBookCatalog
from misesgpt.epub_fetcher import MisesEPUBookCatalog

class MisesDataset:
  def __init__(self, tokenizer, max_length, cache_tokenization=False):
    assert(tokenizer is not None)

    self.tokenizer = tokenizer
    self.sentences = []
    self.max_length = max_length
    self.cache_tokenization = cache_tokenization
    
    self.get_paragraphs(MisesHTMLBookCatalog().books_json)
    self.get_paragraphs(MisesEPUBookCatalog().books_json)

  def process_paragraph(self, p):
    pen = self.tokenizer(p, add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)
    if len(pen) < self.max_length:
      self.sentences.append(p)

  def get_paragraphs(self, books):
    for book in books:
      with multiprocessing.Pool() as pool:
        pool.map(self.process_paragraph, books[book])
        pool.close()

  def __getitem__(self, item):
    if not self.cache_tokenization:
      return self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)

    if isinstance(self.sentences[item], str):
      self.sentences[item] = self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)

    return self.sentences[item]

  def __len__(self):
    return len(self.sentences)
