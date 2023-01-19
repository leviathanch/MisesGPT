'''
 Copyright (c) 2023-2026 David Lanzendörfer <leviathan@libresilicon.com>
 Distributed under the GNU GPL v2. For full terms see the file docs/COPYING.
'''

from os.path import exists
from os import remove

from misesgpt.html_fetcher import MisesHTMLBookCatalog
from misesgpt.epub_fetcher import MisesEPUBookCatalog


def append_data(file, books):
  for book in books:
    for p in books[book]:
      file.write('<|startoftext|>' + p + '<|endoftext|>\n')

df = 'data.txt'
if exists(df):
  remove(df)

with open(df,'a') as file:
  append_data(file, MisesHTMLBookCatalog().books_json)
  append_data(file, MisesEPUBookCatalog().books_json)
  file.close()
