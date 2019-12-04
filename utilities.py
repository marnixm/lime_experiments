import os
import sys
###for saving results e.d.

def printLog(path, *args):
  text = ' '.join([str(arg) for arg in args])
  print(text)
  with open(path, 'a') as log:
    log.write(text + '\n')
