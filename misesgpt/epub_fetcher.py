'''
 Copyright (c) 2023-2026 David Lanzendörfer <leviathan@libresilicon.com>
 Distributed under the GNU GPL v2. For full terms see the file docs/COPYING.
'''

import ebooklib
from ebooklib import epub

from tqdm import tqdm

from bs4 import BeautifulSoup

from misesgpt.abstract_fetcher import MisesAbstractBookCatalog

class MisesEPUBookCatalog(MisesAbstractBookCatalog):

  filters = [
    'cover',
    'title',
    'toc',
    'TOC',
    'content',
    'copyright',
    'Index',
    'index',
    'back',
    'preface',
    'Preface',
    'appen',
    'references',
  ]
  download_base_url = 'https://cdn.mises.org'
  search_filter = "/library/books?book_type=537"

  def __init__(self, json_file='./mises_ebup_books.json', cache_dir='./epub_dir'):
    MisesAbstractBookCatalog.__init__(self, json_file=json_file, cache_dir=cache_dir)

  def find_epub_book(self, formats):
    ret = []
    links = formats.find_all('a')
    for a in links:
      if a.text == "EPUB":
        ret.append(a['href'])
    return ret

  def fetch_epub_books(self, url):
    ret=[]
    req = self.request(self.base_url+url,'html')
    soup = BeautifulSoup(req, 'html.parser')
    books = soup.find_all("div", {"class": "book-formats"})
    for book in books:
      ret += self.find_epub_book(book)
    return ret

  def get_epub_urls(self, page_num=0):
    books=[]
    pages = self.fetch_pagination(self.search_filter+"&page="+str(page_num))
    page_nums = []
    for i in pages:
      i = i.replace(self.search_filter,'').replace('&page=','')
      page_nums.append(1 if i == '' else int(i))
    for i, page in enumerate(pages):
      book = self.fetch_epub_books(page)
      if len(book) > 0:
        if book in books:
          return books
        books += book

    if page_num < max(page_nums):
      books += self.get_epub_urls(page_num=max(page_nums))

    return books

  def paragraph_list(self, chapter):
    ret = []
    soup = BeautifulSoup(chapter.get_body_content(), 'html.parser')
    ps = soup.find_all('p')
    for p in ps:
      pc = p.get_text().replace('\u00a0','').replace('\r','')
      for spc in pc.split('\n'):
        if len(spc)>0:
          ret.append(spc)
    return ret

  def in_filters(self, name):
    for n in self.filters:
      if n in name:
        return True
    return False

  def update(self):
    print("Updating EPUBs")
    urls = self.get_epub_urls()
    for i in tqdm(range(len(urls))):
      url = urls[i]
      if "Spanish" in url:
        continue
      if "Chinese" in url:
        continue
      bname = url.replace(self.download_base_url+'/','').split('.epub')[0]
      if bname in self.books_json:
        continue
      self.books_json[bname]=[]
      book = epub.read_epub(self.request_path(url,'epub'))
      for item in list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT)):
        if self.in_filters(item.get_name()):
          continue
        pl = self.paragraph_list(item)
        if len(pl)>0:
          if not "table of content" in pl[0].lower():
            self.books_json[bname] += pl
    self.save()
