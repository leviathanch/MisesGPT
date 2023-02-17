import sentencepiece as sp

from misesgpt.html_fetcher import MisesHTMLBookCatalog
from misesgpt.epub_fetcher import MisesEPUBookCatalog
from misesgpt.basic_words import MisesBasicWords

from transformers import  ReformerTokenizerFast
from tokenizers import AddedToken

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
    f.write(p+'\n')
  f.close()

#uds = '<mask>'
#for i in range(10):
#  uds += ',<speaker'+str(i)+'>'

sp.SentencePieceTrainer.train(
  input=datafile,
  model_prefix='model/remilamda',
  vocab_size=52_000,
  #user_defined_symbols=uds,
  max_sentence_length=8192,
)

tokenizer = ReformerTokenizerFast(
  vocab_file = 'model/remilamda.model',
  return_special_tokens_mask = True,
  add_special_tokens = True,
  padding = True,
  bos_token = AddedToken('<s>'),
  eos_token = AddedToken('</s>'),
  pad_token = AddedToken('<pad>'),
  unk_token = AddedToken('<unk>'),
  mask_token = AddedToken('<mask>'),
  sep_token = AddedToken('<sep>'),
)
tokenizer.add_special_tokens({"additional_special_tokens": [
  AddedToken("\n")
]})

tokenizer.train_new_from_iterator(paragraphs, 52000)
tokenizer.save_pretrained("remitokenizer")
