import sys
import codecs
from collections import defaultdict

pathDic_ = "C://Users/Q-Bit/Desktop/thWL.txt"

myDic = defaultdict(lambda: 0)

wordLen = defaultdict(lambda: 0)

def readFile():
    with codecs.open(pathDic_, 'r', 'utf-8') as f:
        for line in f:
            line = line.strip(" ")
            line = line.strip("\r\n")
            if (len(line) > 1):
                myDic[line] = 1

    for a,b in sorted(myDic.items()):
        wordLen[len(a)] += 1;

readFile()

print (sorted(wordLen.items()))

s = "โดยทั้งผู้ซื้อและผู้ขายจะมีภาระผูกพันที่จะต้องทำตามสัญญาที่กำหนดไว้"

t = ""

print (s)
for i in range(2,30,1):
    if i <= len(s):
        l = []
        for j in range(0,len(s),1):
            if (myDic[s[j:j+i]] > 0 and len(s[j:j+i]) == i):
                l.append(s[j:j+i])
        print (l)
