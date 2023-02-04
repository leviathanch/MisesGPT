from transformers import ReformerTokenizer, ReformerTokenizerFast

from misesgpt.html_fetcher import MisesHTMLBookCatalog
from misesgpt.epub_fetcher import MisesEPUBookCatalog

html_books = MisesHTMLBookCatalog()
epub_books = MisesEPUBookCatalog()
paragraphs = []
for book in html_books.books_json:
  paragraphs += html_books.books_json[book]
for book in epub_books.books_json:
  paragraphs += epub_books.books_json[book]

for i, p in enumerate(paragraphs):
  paragraphs[i] = '<|startoftext|>' + p + '<|endoftext|>'

# Initialize a tokenizer
tokenizer = ReformerTokenizerFast(
  vocab_file = 'model/spiece.model',
  return_special_tokens_mask = True,
  bos_token = '<|startoftext|>',
  eos_token = '<|endoftext|>',
  pad_token = '<|pad|>',
  unk_token = '<|unk|>',
  mask_token = '<|mask|>',
  padding = True,
)

#help(tokenizer)
new_tokenizer = tokenizer.train_new_from_iterator(paragraphs, 52_000)
# Save files to disk
new_tokenizer.save_vocabulary('model')
