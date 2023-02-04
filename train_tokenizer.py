from transformers import ReformerTokenizer, ReformerTokenizerFast

import sentencepiece as sp

from misesgpt.html_fetcher import MisesHTMLBookCatalog
from misesgpt.epub_fetcher import MisesEPUBookCatalog
from misesgpt.basic_words import MisesBasicWords

datafile = 'data.txt'

word_definitions = MisesBasicWords()
html_books = MisesHTMLBookCatalog()
epub_books = MisesEPUBookCatalog()
paragraphs = []

for word in word_definitions.words:
  paragraphs += word_definitions.words[word]
for book in html_books.books_json:
  paragraphs += html_books.books_json[book]
for book in epub_books.books_json:
  paragraphs += epub_books.books_json[book]

with open(datafile,'w') as f:
  for p in paragraphs:
    f.write('<s>' + p + '</s>\n')
  f.close()

sp.SentencePieceTrainer.train(input=datafile,model_prefix='model/spiece', vocab_size=52_000)

# Initialize a tokenizer
tokenizer = ReformerTokenizer.from_pretrained(
  'model',
  return_special_tokens_mask = True,
  bos_token = '<s>',
  eos_token = '</s>',
  pad_token = '<pad>',
  unk_token = '<unk>',
  mask_token = '<mask>',
  padding = True,
)

tokenizer.save_pretrained('model')
