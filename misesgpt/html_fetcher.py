'''
 Copyright (c) 2023-2026 David Lanzendörfer <leviathan@libresilicon.com>
 Distributed under the GNU GPL v2. For full terms see the file docs/COPYING.
'''

from bs4 import BeautifulSoup
from tqdm import tqdm

from misesgpt.abstract_fetcher import MisesAbstractBookCatalog

class MisesHTMLBookCatalog(MisesAbstractBookCatalog):
  
  search_filter = "/library/books?book_type=538"

  def __init__(self, json_file='./mises_html_books.json', cache_dir='./html_dir'):
    MisesAbstractBookCatalog.__init__(self, json_file=json_file, cache_dir=cache_dir)

  def fetch_table_of_content(self, url):
    ret=[]
    req = self.request(url, 'html')
    soup = BeautifulSoup(req, 'html.parser')
    divs = soup.find_all("div", {"class": "view-grouping-content"})
    for div in divs:
      groups = div.find_all("div", {"class": "view-grouping"})
      for group in groups:
        for a in self.extract_links(group):
          if 'footnote' in a['href']:
            continue
          if a.text not in ['Bibliography', 'Index']:
            ret.append(a['href'])
    return ret

  def fetch_book_page(self, url, depth):
    ret=[]
    req = self.request(url, 'html')
    soup = BeautifulSoup(req, 'html.parser')
    content = soup.find("div", {"class": "book-content"})
    if content is not None:
      for a in content.find_all("a"):
        a.decompose()
      for p in content.find_all("p"):
        for sp in p.text.replace('\t','').split('\n'):
          if len(sp) > 0:
            ret.append(sp)
    elif depth > 0:
      children = soup.find_all("div", {"class": "children"})
      for child in children:
        for a in self.extract_links(child):
          ret += self.fetch_book_page(self.base_url+a['href'], depth-1)
    return ret

  def find_html_book(self, formats):
    ret = []
    for a in formats.find_all('a'):
      if a.text == "HTML Book":
        ret.append(a['href'])
    return ret

  def fetch_html_books(self, url):
    ret=[]
    req = self.request(self.base_url+url, 'html')
    soup = BeautifulSoup(req, 'html.parser')
    books = soup.find_all("div", {"class": "book-formats"})
    for book in books:
      ret += self.find_html_book(book)
    return ret

  def get_html_urls(self, page_num=0):
    books = []
    pages = self.fetch_pagination(self.search_filter+"&page="+str(page_num))
    page_nums = []
    for i in pages:
      i = i.replace(self.search_filter,'').replace('&page=','')
      page_nums.append(1 if i == '' else int(i))
    for i, page in enumerate(pages):
      book = self.fetch_html_books(page)
      if len(book) > 0:
        if book in books:
          return books
        books += book
    if page_num < max(page_nums):
      books += self.get_html_urls(page_num=max(page_nums))

    return books

  def update(self):
    print("Updating HTML books")
    urls = self.get_html_urls()
    for i in tqdm(range(len(urls))):
      content = urls[i]
      if content == '':
        continue
      fname = content.replace("/library/",'').replace("/html/c/",'_')
      if fname in self.books_json:
        continue
      self.books_json[fname] = self.fetch_book_page(content, 3)
    self.save()
