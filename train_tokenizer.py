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

uds = '<mask>'
for i in range(10):
  uds += ',<speaker'+str(i)+'>'

sp.SentencePieceTrainer.train(
  input=datafile,
  model_prefix='model/remilamda',
  vocab_size=52_000,
  bos_id=0,
  pad_id=1,
  eos_id=2,
  unk_id=3,
  user_defined_symbols=uds,
)
