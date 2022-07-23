'''
Author: liupengfei
Function: count lines of code in a folder iteratively
Shell-format: cmd [dir]
Attention: default file encode is utf8 and default file type is java-source-file. But users can customize this script by just modifing global variables.
'''
import sys
import os
import codecs
from _pyio import open
totalCount = 0
fileType = '.py'
descLineBegin = '//'
descBlockBegin = r'/**'
descBlockEnd = r'*/'
fileEncode = 'utf-8'
def main():
  DIR = os.getcwd()
  if len(sys.argv) >= 2:
    DIR = sys.argv[1]
  if os.path.exists(DIR) and os.path.isdir(DIR):
    print('target directory is %s' % DIR)
    countDir(DIR)
    print('total code line is %d' % totalCount)
  else:
    print('target should be a directory!')
def isFileType(file):
  return len(fileType) + file.find(fileType) == len(file)
def countDir(DIR):
  for file in os.listdir(DIR):
    absPath = DIR + os.path.sep + file
    if os.path.exists(absPath):
      if os.path.isdir(absPath):
        countDir(absPath)
      elif isFileType(absPath):
        try:
          countFile(absPath)
        except UnicodeDecodeError:
          print(
            '''encode of %s is different, which
is not supported in this version!'''
            )
def countFile(file):
  global totalCount
  localCount = 0
  isInBlockNow = False
  f = codecs.open(file, 'r', fileEncode)
  for line in f:
    if (not isInBlockNow) and line.find(descLineBegin) == 0:
      pass
    elif (not isInBlockNow) and line.find(descBlockBegin) >= 0:
      if line.find(descBlockBegin) > 0:
        localCount += 1
      isInBlockNow = True
    elif isInBlockNow and line.find(descBlockEnd) >= 0:
      if line.find(descBlockEnd) + len(descBlockEnd) < len(line):
        localCount += 1
      isInBlockNow = False
    elif (not isInBlockNow) and len(line.replace('\\s+', '')) > 0:
      localCount += 1
  f.close()
  totalCount += localCount
  print('%s : %d' % (file, localCount))
if __name__ == '__main__':
  main()