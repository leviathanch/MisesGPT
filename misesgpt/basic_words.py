'''
 Copyright (c) 2023-2026 David Lanzendörfer <leviathan@libresilicon.com>
 Distributed under the GNU GPL v2. For full terms see the file docs/COPYING.
'''

import csv

class MisesBasicWords:

  words = {}
  
  def __init__(self):
    # Word,Meaning,Examples/0,Examples/1,Examples/2,Examples/3,Examples/4,Examples/5,Examples/6,Examples/7,Examples/8,Examples/9
    with open('word-meaning-examples.csv') as f:
      c = csv.reader(f)
      for l in c:
        if len(l)>1:
          self.words[l[0]]=[]
        if len(l)>2:
          self.words[l[0]].append(l[0]+': '+l[1])
        for i in range(2,12):
          if len(l)>i+1:
            if len(l[i])>0:
              self.words[l[0]].append(l[i])
      f.close()
