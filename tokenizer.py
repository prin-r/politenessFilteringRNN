import sys
import numpy as np
import random
from random import randint
from random import randrange, sample
import codecs
from collections import defaultdict

path_ = "C://Users/Q-Bit/Desktop/testPyData/goodData.txt"
path2_ = "C://Users/Q-Bit/Desktop/testPyData/goodSubSet.txt"

path3_ = "C://Users/Q-Bit/Desktop/testPyData/bw.txt"
path4_ = "C://Users/Q-Bit/Desktop/testPyData/bw2.txt"

path5_ = "C://Users/Q-Bit/Desktop/testPyData/badSentences.txt"

path6_ = "C://Users/Q-Bit/Desktop/testPyData/subset/trainBadSub.txt"
path7_ = "C://Users/Q-Bit/Desktop/testPyData/subset/trainGoodSub.txt"

path8_ = "C://Users/Q-Bit/Desktop/testPyData/subset/testBadSub.txt"
path9_ = "C://Users/Q-Bit/Desktop/testPyData/subset/testGoodSub.txt"

path10_ = "C://Users/Q-Bit/Desktop/testPyData/shortenSentence/shortenGood.txt"
path11_ = "C://Users/Q-Bit/Desktop/testPyData/shortenSentence/shortenBad.txt"

path12_ = "C://Users/Q-Bit/Desktop/testPyData/shortenSentence/sGoodTrain.txt"
path13_ = "C://Users/Q-Bit/Desktop/testPyData/shortenSentence/sGoodTest.txt"

path14_ = "C://Users/Q-Bit/Desktop/testPyData/shortenSentence/sBadTrain.txt"
path15_ = "C://Users/Q-Bit/Desktop/testPyData/shortenSentence/sBadTest.txt"

maxSample = 1500000

dic = defaultdict(lambda: 0)

def random_insert(lst, item):
    lst.insert(randrange(len(lst)+1), item)

def maxVal(a,b):
    if (a > b):
        return a
    return b

def minVal(a,b):
    if (a <= b):
        return a
    return b

def makeDic():
    with codecs.open(path_, 'r', 'utf-8') as f:
        i = 0
        for line in f:
            ls = line.rstrip()
            chars = [x.replace("\r\n", "") for x in ls]
            chars = [x.replace(" ", "") for x in ls]
            i += 1
            if (i % 5000 == 0):
                print (i)
            for x in chars:
                dic[x] += 1

def makeBadSentences():
    with codecs.open(path2_, 'r', 'utf-8') as f:
        bwSet = []
        with codecs.open(path4_, 'r', 'utf-8') as bwF:
            for line in bwF:
                ls = line.rstrip()
                chars = ls.replace("\r\n", "")
                bwSet.append(chars)

        lenBwSet = len(bwSet)
        print (lenBwSet)
        with codecs.open(path5_, 'w', 'utf-8') as out:
            j = 0
            for line in f:
                line = line.rstrip()
                words = line.split(" ")
                lw = len(words)
                n_nw = randint(1, 3)

                if (n_nw > lw):
                    n_nw = lw

                if (n_nw <= 0):
                    n_nw = 1

                """print (n_nw)
                print(lw)
                print(words)"""

                n_nw_list = []
                for i in range(n_nw):
                    if lw-1 > 0:
                        n_nw_list.append(randint(0,lw-1))
                    else:
                        n_nw_list.append(0)
                #print(n_nw_list)
                for e in n_nw_list:
                    words.insert(e,bwSet[randint(0,lenBwSet-1)])
                s = ""
                for e in words:
                    s += e

                j += 1
                if (j % 10000 == 0):
                    print(j)
                    print(s)
                out.write(s + "\n")

def noRepeatSentence():
    with codecs.open(path3_, 'r', 'utf-8') as f:
        i = 0
        for line in f:
            i += 1
            line = line.rstrip()
            line = line.replace("\ufeff" , "")
            words = line.split(" ")
            s = ""
            for word in words:
                s += word
            dic[s] = line
            if (i % 10000 == 0):
                print(i)
                print(words)

    with codecs.open(path4_, 'w', 'utf-8') as f2:
        i = 0
        for a, b in sorted(dic.items()):
            i += 1
            if (i % 10000 == 0):
                print(i)
                print(b)
            f2.write(b + "\n")

def makeSub():
    lines = []
    i = 0
    with codecs.open(path11_, 'r', 'utf-8') as f:
        lines = []
        for line in f:
            s = line.replace(" ", "")
            lines.append(s)
            i += 1
            if i % 10000 == 0:
                print (i)
    random.shuffle(lines)
    print (len(lines))
    i = 0
    with codecs.open(path14_, 'w', 'utf-8') as f1:
        for line in lines:
            f1.write(line)
            i += 1
            if i > maxSample:
                break
            if i % 10000 == 0:
                print (i)

    with codecs.open(path15_, 'w', 'utf-8') as f2:
        lines = lines[maxSample + 1:-1]
        for line in lines:
            f2.write(line)
            i += 1
            if i > maxSample + 10000:
                break
            if i % 1000 == 0:
                print(i)

def countSentenceSize():
    with codecs.open(path11_, 'r', 'utf-8') as f:
        i = 0
        for line in f:
            ls = line.rstrip()
            chars = [x.replace("\r\n", "") for x in ls]
            chars = [x.replace(" ", "") for x in ls]
            dic[len(chars)] += 1
            i += 1
            if (len(chars) == 0):
                print (line)
                print (i)
                print("__________________________")
            if i % 10000 == 0:
                print(chars)
                print(i)

def shortenAndBadSentence():
    bwSet = set()
    with codecs.open(path4_, 'r', 'utf-8') as bwF:
        for line in bwF:
            ls = line.rstrip()
            chars = ls.replace("\r\n", "")
            bwSet.add(chars)

    lenBwSet = len(bwSet)
    print (bwSet)

    with codecs.open(path2_, 'r', 'utf-8') as f:
        with codecs.open(path11_, 'w', 'utf-8') as out:
            i = 0
            for line in f:
                ls = line.rstrip()
                lsNoRN = ls.replace("\r", "")
                lsNoRN = lsNoRN.replace("\n", "")

                lsNoSpace = lsNoRN.replace(" ", "")
                chars = [x for x in lsNoSpace]
                words = lsNoRN.split(" ")

                s1 = ""
                s2 = ""
                s3 = ""
                s4 = ""

                if len(chars) > 150:
                    s1 = ""
                    s2 = ""
                    s3 = ""
                    s4 = ""

                    ws1 = []
                    ws2 = []
                    ws3 = []
                    ws4 = []

                    for w in words:
                        if len(s1 + w) < 50:
                            s1 += w
                            ws1.append(w)
                        else:
                            if len(s2 + w) < 50:
                                s2 += w
                                ws2.append(w)
                            else:
                                if len(s3 + w) < 50:
                                    s3 += w
                                    ws3.append(w)
                                else :
                                    if len(s4 + w) < 50:
                                        s4 += w
                                        ws4.append(w)

                    maxBw = randint(1, 3)
                    ii = 0
                    ss = random.sample(bwSet, 1)[0]
                    while (ii < maxBw and len(s1 + ss) < 100):
                        random_insert(ws1, ss)
                        s1 += ss
                        ss = random.sample(bwSet, 1)[0]
                        ii += 1

                    maxBw = randint(1, 3)
                    ii = 0
                    ss = random.sample(bwSet, 1)[0]
                    while (ii < maxBw and len(s2 + ss) < 100):
                        random_insert(ws2, ss)
                        s2 += ss
                        ss = random.sample(bwSet, 1)[0]
                        ii += 1

                    maxBw = randint(1, 3)
                    ii = 0
                    ss = random.sample(bwSet, 1)[0]
                    while (ii < maxBw and len(s3 + ss) < 100):
                        random_insert(ws3, ss)
                        s3 += ss
                        ss = random.sample(bwSet, 1)[0]
                        ii += 1

                    maxBw = randint(1, 3)
                    ii = 0
                    ss = random.sample(bwSet, 1)[0]
                    while (ii < maxBw and len(s4 + ss) < 100):
                        random_insert(ws4, ss)
                        s4 += ss
                        ss = random.sample(bwSet, 1)[0]
                        ii += 1

                    s1 = ""
                    for w in ws1:
                        s1 += w

                    s2 = ""
                    for w in ws2:
                        s2 += w

                    s3 = ""
                    for w in ws3:
                        s3 += w

                    s4 = ""
                    for w in ws4:
                        s4 += w

                    out.write(s1.replace("\n", "") + "\n")
                    out.write(s2.replace("\n", "") + "\n")
                    out.write(s3.replace("\n", "") + "\n")
                    out.write(s4.replace("\n", "") + "\n")

                elif  len(chars) > 100:
                    s1 = ""
                    s2 = ""
                    s3 = ""

                    ws1 = []
                    ws2 = []
                    ws3 = []

                    for w in words:
                        if len(s1 + w) < 50:
                            s1 += w
                            ws1.append(w)
                        else:
                            if len(s2 + w) < 50:
                                s2 += w
                                ws2.append(w)
                            else:
                                if len(s3 + w) < 50:
                                    s3 += w
                                    ws3.append(w)

                    maxBw = randint(1, 3)
                    ii = 0
                    ss = random.sample(bwSet, 1)[0]
                    while (ii < maxBw and len(s1 + ss) < 100):
                        random_insert(ws1, ss)
                        s1 += ss
                        ss = random.sample(bwSet, 1)[0]
                        ii += 1

                    maxBw = randint(1, 3)
                    ii = 0
                    ss = random.sample(bwSet, 1)[0]
                    while (ii < maxBw and len(s2 + ss) < 100):
                        random_insert(ws2, ss)
                        s2 += ss
                        ss = random.sample(bwSet, 1)[0]
                        ii += 1

                    maxBw = randint(1, 3)
                    ii = 0
                    ss = random.sample(bwSet, 1)[0]
                    while (ii < maxBw and len(s3 + ss) < 100):
                        random_insert(ws3, ss)
                        s3 += ss
                        ss = random.sample(bwSet, 1)[0]
                        ii += 1

                    s1 = ""
                    for w in ws1:
                        s1 += w

                    s2 = ""
                    for w in ws2:
                        s2 += w

                    s3 = ""
                    for w in ws3:
                        s3 += w

                    out.write(s1.replace("\n", "") + "\n")
                    out.write(s2.replace("\n", "") + "\n")
                    out.write(s3.replace("\n", "") + "\n")

                elif len(chars) > 50:
                    s1 = ""
                    s2 = ""

                    ws1 = []
                    ws2 = []

                    for w in words:
                        if len(s1 + w) < 50:
                            s1 += w
                            ws1.append(w)
                        else :
                            if len(s2 + w) < 50:
                                s2 += w
                                ws2.append(w)

                    maxBw =  randint(1,3)
                    ii = 0
                    ss = random.sample(bwSet, 1)[0]
                    while (ii < maxBw and len(s1 + ss) < 100):
                        random_insert(ws1,ss)
                        s1 += ss
                        ss = random.sample(bwSet, 1)[0]
                        ii += 1

                    maxBw = randint(1, 3)
                    ii = 0
                    ss = random.sample(bwSet, 1)[0]
                    while (ii < maxBw and len(s2 + ss) < 100):
                        random_insert(ws2, ss)
                        s2 += ss
                        ss = random.sample(bwSet, 1)[0]
                        ii += 1

                    s1 = ""
                    for w in ws1:
                        s1 += w

                    s2 = ""
                    for w in ws2:
                        s2 += w

                    out.write(s1.replace("\n", "") + "\n")
                    out.write(s2.replace("\n", "") + "\n")

                else :
                    for w in words:
                        s1 += w

                    maxBw =  randint(1,3)
                    ii = 0
                    ss = random.sample(bwSet, 1)[0]
                    while (ii < maxBw and len(s1 + ss) < 100):
                        random_insert(words,ss)
                        s1 += ss
                        ss = random.sample(bwSet, 1)[0]
                        ii += 1

                    s1 = ""
                    for w in words:
                        s1 += w
                    out.write(s1.replace("\n", "") + "\n")

                i += 1
                if i % 10000 == 0:
                    print(s1+"\n")
                    print(s2+"\n")
                    print(s3 + "\n")
                    print(s4 + "\n")
                    print(i)


def shortenSentence():
    with codecs.open(path2_, 'r', 'utf-8') as f:
        with codecs.open(path10_, 'w', 'utf-8') as out:
            i = 0
            for line in f:
                ls = line.rstrip()
                lsNoRN = ls.replace("\r\n", "")

                lsNoSpace = lsNoRN.replace(" ", "")
                chars = [x for x in lsNoSpace]
                s1 = ""
                s2 = ""
                if len(chars) > 100:
                    words = line.split(" ")
                    exceed = False
                    for w in words:
                        if not exceed and len(s1 + w) > 100:
                            exceed = True

                            if not exceed:
                                s1 += w
                            else:
                                s2 += w

                            if (len(s1) > 0):
                                out.write(s1.replace("\n", "") + "\n")
                            if (len(s2) > 0):
                                out.write(s2.replace("\n", "") + "\n")
                        else:
                            if (len(lsNoSpace) > 0):
                                out.write(lsNoSpace.replace("\n", "") + "\n")

                            i += 1
                            if i % 10000 == 0:
                                if len(chars) > 100:
                                     print(s1 + "\n")
                                     print(s2 + "\n")
                                else:
                                    print(lsNoSpace + "\n")
                                print(i)


#makeBadSentences()
#noRepeatSentence()
#makeSub()
#shortenSentence()
#shortenAndBadSentence()
#countSentenceSize()

for a,b in sorted(dic.items()):
    print("%s --> %r" % (a,b))

print(len(dic))