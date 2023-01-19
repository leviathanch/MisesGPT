'''
 Copyright (c) 2023-2026 David Lanzendörfer <leviathan@libresilicon.com>
 Distributed under the GNU GPL v2. For full terms see the file docs/COPYING.
'''

import hashlib
from os.path import exists, join
from os import mkdir
import json
import requests

from bs4 import BeautifulSoup

class MisesAbstractBookCatalog:
  books_json = {}
  base_url = "https://mises.org"

  def __init__(self, json_file='./mises_books.json', cache_dir='./cache'):

    self.json_file = json_file
    self.cache_dir = cache_dir

    if exists(self.json_file):
      with open(self.json_file,'r') as file:
        self.books_json = json.load(file)
        file.close()
    
    if not exists(self.cache_dir):
      mkdir(self.cache_dir)

  def request(self, url, suffix):
    ret = ''
    fn = join(self.cache_dir, hashlib.sha256(url.encode()).hexdigest() + '.' + suffix)
    if exists(fn):
      with open(fn,'rb') as f:
        ret = f.read()
        f.close()
    else:
      req = requests.get(url)
      ret = req.content
      with open(fn,'wb') as f:
        f.write(ret)
        f.close()
    return ret

  def request_path(self, url, suffix):
    fn = join(self.cache_dir, hashlib.sha256(url.encode()).hexdigest() + '.' + suffix)
    self.request(url, suffix)
    return fn

  def extract_links(self, group):
    ret=[]
    for a in group.find_all("a", attrs = {'href' : True}):
      if "http://" in a['href'] or "https://" in a['href']:
        continue
      ret.append(a)
    return ret

  def fetch_pagination(self, url):
    ret=[]
    req = self.request(self.base_url+url, 'html')
    soup = BeautifulSoup(req, 'html.parser')
    divs = soup.find("ul", {"class": "pagination"})
    for li in divs.find_all("li")[1:-1]:
      a = self.extract_links(li)
      if len(a)>0:
        a = a[0]
        if '#' not in a['href']:
          ret.append(a['href'])
    return ret

  def save(self):
    with open(self.json_file,'w') as file:
        file.write(json.dumps(self.books_json, indent=4))
        file.close()
