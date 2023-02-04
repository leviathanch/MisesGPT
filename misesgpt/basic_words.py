'''
 Copyright (c) 2023-2026 David Lanzendörfer <leviathan@libresilicon.com>
 Distributed under the GNU GPL v2. For full terms see the file docs/COPYING.
'''

import csv

class MisesBasicWords:

  words = {}
  
  def __init__(self):
    with open('word-meaning-examples.csv') as f:
      c = csv.reader(f) #, delimiter=',', quotechar='|')
      for l in c:
        if len(l)>1:
          self.words[l[0]]=[]
        if len(l)>2:
          self.words[l[0]].append(l[0]+': '+l[1])
        for i in range(2,20):
          if len(l)>i+1:
            if len(l[i])>0:
              self.words[l[0]].append(l[i])
      f.close()

