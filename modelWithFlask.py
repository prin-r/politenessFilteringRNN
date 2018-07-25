import sys
import tensorflow as tf
import numpy as np
import codecs
import random
from collections import defaultdict

from flask import Flask,redirect
from flask import request, jsonify, json, make_response, current_app
from flask_cors import CORS, cross_origin

from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={r"/test": {"origins": "http://localhost:5000"}})

# odd or even
dicEncode = defaultdict(lambda: 0)
dicDecode = defaultdict(lambda: 0)

path_ = "C://Users/Q-Bit/Desktop/testPyData/data.txt"

pathDicEnc_ = "C://Users/Q-Bit/Desktop/testPyData/dicEnc.txt"

pathGoodTrain_ = "C://Users/Q-Bit/Desktop/testPyData/shortenSentence/sGoodTrain.txt"
pathGoodTest_ = "C://Users/Q-Bit/Desktop/testPyData/shortenSentence/sGoodTest.txt"

pathBadTrain_ = "C://Users/Q-Bit/Desktop/testPyData/shortenSentence/sBadTrain.txt"
pathBadTest_ = "C://Users/Q-Bit/Desktop/testPyData/shortenSentence/sBadTest.txt"

savePath_ = "C://Users/Q-Bit/Desktop/testPyData/save/"

writeSentencesCollectionPath_ = "C://Users/Q-Bit/Desktop/testPyData/web/sentencesCollection"

myWebUrl_ =  "http://127.0.0.1:5000/"
#"https://38b28dc5.ngrok.io/"

trainingSet = []
testSet = []
batchTrain = []
batchTest = []

trainingSet_token = []
testSet_token = []

n_hidden_ = 128
vocab_size_ = 3
sequenceLength_ = 40
outPutSize_ = 2
epoch_ = 1000000
batch_size_ = 128
num_layers_ = 2


def readFile():
    with codecs.open(pathGoodTrain_, 'r', 'utf-8') as f:
        i = 0
        j = 0
        for line in f:
            line1 = (line.strip()).replace("\n", "")
            line1 = line1.replace("\r", "")
            trainingSet.append([line1, [1, 0]])  # [good, bad]
            i += 1
            if i == 10000:
                i = 0
                j += 10000
                print(j)

    with codecs.open(pathBadTrain_, 'r', 'utf-8') as f:
        for line in f:
            line1 = (line.strip()).replace("\n", "")
            line1 = line1.replace("\r", "")
            trainingSet.append([line1, [0, 1]])  # [good, bad]

    with codecs.open(pathGoodTest_, 'r', 'utf-8') as f:
        i = 0
        j = 0
        for line in f:
            line1 = (line.strip()).replace("\n", "")
            line1 = line1.replace("\r", "")
            testSet.append([line1, [1, 0]])  # [good, bad]
            i += 1
            if i == 10000:
                i = 0
                j += 10000
                print(j)

    with codecs.open(pathBadTest_, 'r', 'utf-8') as f:
        for line in f:
            line1 = (line.strip()).replace("\n", "")
            line1 = line1.replace("\r", "")
            testSet.append([line1, [0, 1]])  # [good, bad]


def tokenization():
    for e in trainingSet:
        for ee in e[0]:
            dicEncode[ee] += 1

    for e in testSet:
        for ee in e[0]:
            dicEncode[ee] += 1

    startIndex = 1
    for key in dicEncode:
        dicEncode[key] = startIndex
        startIndex += 1

    with codecs.open(pathDicEnc_, 'w', 'utf-8') as f:
        for a, b in sorted(dicEncode.items()):
            f.write(a + " " + str(b) + "\n")

    print("shuffle...")
    random.shuffle(trainingSet)
    random.shuffle(testSet)

    for e in trainingSet:
        trainingSet_token.append([[dicEncode[ee] for ee in e[0]], e[1]])

    for e in testSet:
        testSet_token.append([[dicEncode[ee] for ee in e[0]], e[1]])

    for a, b in sorted(dicEncode.items()):
        dicDecode[b] = a


def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]


def makeBatch():
    for i in range(0,5):
        batchTrain.append([])
        batchTest.append([])

    vocab_size_ = len(dicEncode) + 1
    print (vocab_size_)
    makeBatchCount = 0

    for e in trainingSet_token:
        if len(e[0]) <= 20:
            while len(e[0]) < 20:
                e[0].append(0)
            batchTrain[0].append(e)
        elif len(e[0]) <= 40:
            while len(e[0]) < 40:
                e[0].append(0)
            batchTrain[1].append(e)
        elif len(e[0]) <= 60:
            while len(e[0]) < 60:
                e[0].append(0)
            batchTrain[2].append(e)
        elif len(e[0]) <= 80:
            while len(e[0]) < 80:
                e[0].append(0)
            batchTrain[3].append(e)
        else:
            while len(e[0]) < 100:
                e[0].append(0)
            batchTrain[4].append(e)

        makeBatchCount += 1
        if makeBatchCount % 10000 == 0:
            print("makeBatchCount %d" % makeBatchCount)

    batchTrainCount = 0
    for bt in batchTrain:
        batchTrain[batchTrainCount] = list(chunks(bt, batch_size_))
        batchTrainCount += 1


def decodeToString(x):  # x have to be an array of int such as [1 ,2 , 3]
    s = ""
    for e in x:
        if dicDecode[e] != 0:
            s += dicDecode[e]
    return s


# -----------------------------------------------------------------------------------------------------------------------

def genInput(vs, bs, sl):
    return np.random.randint(vs, size=(bs, sl))


def genTarget(input, bs, os):
    a = np.sum(input, axis=1) % outPutSize_
    tar = np.zeros((bs, os))
    tar[np.arange(bs), a] = 1
    return tar


def oneHot(input, bs, sl, vs):
    oH = np.zeros((bs, sl, vs))
    for i in range(0, bs):
        oH[i][np.arange(sl), input[i]] = 1
    return oH

def vectorOfWord2OneHot(x):
    inputList = []
    outputList = []
    for e in x:
        #print (e)
        l = len(e[0])
        b = np.zeros((l, vocab_size_))
        b[np.arange(l), e[0]] = 1
        inputList.append(b)
        outputList.append(e[1])

    return np.array(inputList),np.array(outputList)


def training():
    data = tf.placeholder(tf.float32, [None, None, vocab_size_], name='data') #data = tf.placeholder(tf.float32, [None, sequenceLength_,vocab_size_]) #[Batch Size, Sequence Length, Input Dimension]
    target = tf.placeholder(tf.float32, [None, outPutSize_], name='target')
    dropout = tf.placeholder(tf.float32,name='dropout')

    weight0 = tf.Variable(tf.truncated_normal([vocab_size_,vocab_size_]),name='weight0')
    bias0 = tf.Variable(tf.constant(0.1, shape=[vocab_size_]),name='bias0')
    y = tf.reshape(tf.matmul(tf.reshape(data, [-1, vocab_size_]), weight0) + bias0, tf.shape(data))

    cells = []
    for _ in range(num_layers_):
      cell = tf.contrib.rnn.GRUCell(n_hidden_)  # Or LSTMCell(num_units)
      cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout)
      cells.append(cell)
    cell = tf.contrib.rnn.MultiRNNCell(cells)

    val, state = tf.nn.dynamic_rnn(cell, y, dtype=tf.float32, time_major=False)

    val = tf.transpose(val, [1, 0, 2])
    last = tf.gather(val, tf.shape(val)[0] - 1)

    weight = tf.Variable(tf.truncated_normal([n_hidden_, outPutSize_]),name='weight1')
    bias = tf.Variable(tf.constant(0.1, shape=[outPutSize_]),name='bias1')

    prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

    argMaxP = tf.argmax(prediction, 1)
    argMaxT = tf.argmax(target, 1)

    diffFromTarget = tf.reduce_sum(tf.abs(argMaxP - argMaxT))

    cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))

    optimizer = tf.train.AdamOptimizer()
    minimize = optimizer.minimize(cross_entropy)

    mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
    error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)

    dropOutVal = 0.5

    len20 = len(batchTrain[0])
    len40 = len(batchTrain[1])
    len60 = len(batchTrain[2])
    len80 = len(batchTrain[3])
    len100 = len(batchTrain[4])

    for i in range(epoch_):

        #oh, tg = vectorOfWord2OneHot([[[0, 1, 2], [0, 1]], [[1, 0, 1], [0, 0]], [[2, 1, 2], [1, 0]], [[0, 2, 0], [1, 1]]])

        oh20, tg20 = vectorOfWord2OneHot(batchTrain[0][i % len20])
        oh40, tg40 = vectorOfWord2OneHot(batchTrain[1][i % len40])
        oh60, tg60 = vectorOfWord2OneHot(batchTrain[2][i % len60])
        oh80, tg80 = vectorOfWord2OneHot(batchTrain[3][i % len80])
        oh100, tg100 = vectorOfWord2OneHot(batchTrain[4][i % len100])

        sess.run(minimize, {data: oh20, target: tg20, dropout: dropOutVal})
        sess.run(minimize, {data: oh40, target: tg40, dropout: dropOutVal})
        sess.run(minimize, {data: oh60, target: tg60, dropout: dropOutVal})
        sess.run(minimize, {data: oh80, target: tg80, dropout: dropOutVal})
        sess.run(minimize, {data: oh100, target: tg100, dropout: dropOutVal})

        if (i % 200 == 0):
            ts = random.sample(testSet_token, 1000)

            print("_______________________________________________")

            amp = []
            amt = []
            errorAcc = 0
            for e in ts:
                oh1, tg1 = vectorOfWord2OneHot([e])

                amp.append(np.array(sess.run(argMaxP,{data: oh1, target: tg1,dropout: dropOutVal}))[0])
                amt.append(np.array(sess.run(argMaxT, {data: oh1, target: tg1,dropout: dropOutVal}))[0])
                #print(sess.run(diffFromTarget, {data: oh1, target: tg1,dropout: dropOutVal}))
                errorAcc += sess.run(error,{data: oh1, target: tg1,dropout: dropOutVal})

            print(amp)
            print(amt)
            print(np.abs(np.array(amp) - np.array(amt)))

            print("Epoch - ", str(i))
            print('Epoch {:2d} error {:3.1f}%'.format(i, errorAcc * 0.1))
            print('dropOutVal' + str(dropOutVal))
            dropOutVal -= 0.0075
            if (dropOutVal < 0):
                dropOutVal = 0

            saver0 = tf.train.Saver()
            saver0.save(sess, savePath_ + 'my-model-' + str(i))
            # Generates MetaGraphDef.
            # saver0.export_meta_graph(savePath_ + 'my-model-' + str(i) + '.meta')

    sess.close()

def testModel():
    print("testOnly")
    print("loadDicEnc")
    with codecs.open(pathDicEnc_, 'r', 'utf-8') as f:
        ii = 0
        for line in f:
            line1 = line.replace("\n", "")
            words = line1.split(" ")
            print (ii)
            print (line1)
            print (words)
            if len(words) > 0:
                dicEncode[words[0]] = int(words[-1])

    with tf.Session() as sess:

        data = tf.placeholder(tf.float32, [None, None, vocab_size_],name='data')
        target = tf.placeholder(tf.float32, [None, outPutSize_], name='target')
        dropout = tf.placeholder(tf.float32, name='dropout')

        weight0 = tf.Variable(tf.truncated_normal([vocab_size_, vocab_size_]), name='weight0')
        bias0 = tf.Variable(tf.constant(0.1, shape=[vocab_size_]), name='bias0')
        y = tf.reshape(tf.matmul(tf.reshape(data, [-1, vocab_size_]), weight0) + bias0, tf.shape(data))

        cells = []
        for _ in range(num_layers_):
            cell = tf.contrib.rnn.GRUCell(n_hidden_)  # Or LSTMCell(num_units)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)

        val, state = tf.nn.dynamic_rnn(cell, y, dtype=tf.float32, time_major=False)

        val = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val, tf.shape(val)[0] - 1)

        weight = tf.Variable(tf.truncated_normal([n_hidden_, outPutSize_]), name='weight1')
        bias = tf.Variable(tf.constant(0.1, shape=[outPutSize_]), name='bias1')

        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

        argMaxP = tf.argmax(prediction, 1)
        argMaxT = tf.argmax(target, 1)

        diffFromTarget = tf.reduce_sum(tf.abs(argMaxP - argMaxT))

        cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))

        optimizer = tf.train.AdamOptimizer()
        minimize = optimizer.minimize(cross_entropy)

        mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
        error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        iteration = 11600 #13400
        dropOutVal = 0

        #saver = tf.train.import_meta_graph(savePath_ + 'my-model-' + str(iteration) + '.meta', clear_devices=True)
        saver = tf.train.Saver()
        saver.restore(sess, savePath_ + 'my-model-' + str(iteration))

        sentenceCollection = defaultdict(lambda : 0)
        sentence2Prop = []

        #print(sess.run(bias0))

        myList = [set(), set()]
        predDict = defaultdict(lambda : 0)

        def pred(s):
            s_token = []
            for e in s:
                s_token.append(dicEncode[e])

            oh1, tg1 = vectorOfWord2OneHot([[s_token, [0, 0]]])
            return sess.run(prediction, {data: oh1, dropout: dropOutVal})

        @app.route('/test', methods=['POST'])
        @cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
        def check_rudeness():
            data = request.get_json()
            if 'sentence' in data:
                s = data['sentence']
                print(s)
                if s == "writeFile":
                    fName = (str(datetime.now())[:10]).replace(" ", "")
                    fName = fName.replace("-", "")
                    fName = fName.replace(":", "")
                    fName = fName.replace(".", "")
                    with codecs.open(writeSentencesCollectionPath_ + "/Good" + fName + '.txt', 'a', 'utf-8') as out:
                        for e in myList[0]:
                            out.write(str(e) + "\n")
                    with codecs.open(writeSentencesCollectionPath_ + "/Bad" + fName + '.txt', 'a', 'utf-8') as out:
                        for e in myList[1]:
                            out.write(str(e) + "\n")
                    return jsonify(results={'conclusion': 'writeFile', 'obscenity': '0', 'politeness': '0'})

                pd = None
                if s not in predDict.keys():
                    pd = pred(s)
                    predDict[s] = pd
                else :
                    pd = predDict[s]

                print(pd)
                percentOP = [pd[0][0] * 100 / (pd[0][0] + pd[0][1]), pd[0][1] * 100 / (pd[0][0] + pd[0][1])]
                conclusion = 'obscenity' if pd[0][0] < pd[0][1] else 'politeness'
                machineSays = {'conclusion': conclusion, 'obscenity': percentOP[1], 'politeness': percentOP[0]}

                return jsonify(results=machineSays)

            noInputSentence = {'Error': 'cant find key name "sentence"'}
            return jsonify(results=noInputSentence)

        @app.route('/labeling', methods=['POST'])
        @cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
        def labeling():
            data = request.get_json()
            if 'sentence' in data:
                if 'label' in data:
                    if data['label'] == 'politeness':
                        if data['sentence'] in myList[1]:
                            myList[1].remove(data['sentence'])
                        myList[0].add(data['sentence'])
                    elif data['label'] == 'obscenity':
                        if data['sentence'] in myList[0]:
                            myList[0].remove(data['sentence'])
                        myList[1].add(data['sentence'])
            print("*-----------------------------------------------*")
            #print(myList[0])
            #print(myList[1])
            pList = []
            oList = []
            for e in myList[0]:
                pe = predDict[e]
                if pe[0][0] < pe[0][1]:
                    pList.append([e, 'o'])
                else :
                    pList.append([e, 'p'])
            for e in myList[1]:
                pe = predDict[e]
                if pe[0][0] < pe[0][1]:
                    oList.append([e, 'o'])
                else :
                    oList.append([e, 'p'])

            #print (predDict)
            return jsonify(results={'politeness': pList, 'obscenity': oList})

        @app.route('/index')
        def index():
            return '''
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
                        <script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.4/jquery.min.js"></script>
                    </head>
                    <body>
                        
                        <center>
                            <div>
                                <h1>
                                    Please type in a sentence or phrase. (support only in Thai language for now)
                                    <br>
                                    <input type="text" id="sentence" value="" size="60">
                                    <button onclick="sendSentence();">Determine the value of politeness/obscenity</button>
                                </h1>
                            </div>
                            <br>
                            <div>
                                <h2>
                                    <p id="politeness">politeness : not determined yet</p>
                                    <p id="obscenity">obscenity : not determined yet</p>
                                    <p id="conclusion">conclusion : not determined yet</p>
                                </h2>
                            </div>
                        </center>
                        <br>
                        <div style="width: 80%; margin: auto;">
                            <div>
                                <b id="numPolite" style="float: left"></b>
                                <b id="numObsence" style="float: right"></b>
                            </div>
                            <br>
                            <div id="politenessList" style="float:left; width: 50%; background-color: lightblue;">
                            </div>
                            
                            <div id="obscenityList" style="float:right; width: 50%; background-color: pink;">
                            </div>			
                        </div>
                        <script>
                            var lastestSentence = ""
                            var lastestDetermination = ""
                            var lastestPrediction = ""
                            
                            function sendSentence() {
                                lastestDetermination = ""
                                lastestDetermination = ""
                                lastestSentence = document.getElementById("sentence").value
                                $.ajax({
                                    url: 'http://127.0.0.1:5000/test',
                                    contentType: 'application/json;charset=UTF-8',
                                    method: 'POST',
                                    dataType: "html",
                                    data: JSON.stringify({'sentence': lastestSentence}),
                                    success: function (response) {
                                        console.log(response);
                                        updateResult(JSON.parse(response).results);
                                    }
                                })
                            }
                            
                            function sendLabel(det) {
                                if (det == false) {
                                    if (lastestPrediction == 'obscenity') {
                                        lastestDetermination = 'politeness'
                                    }
                                    else if (lastestPrediction == 'politeness') {
                                        lastestDetermination = 'obscenity'
                                    }
                                    else {
                                        lastestDetermination = ""
                                        lastestPrediction = ""
                                    }
                                }
                                else {
                                    if (lastestPrediction == 'obscenity') {
                                        lastestDetermination = 'obscenity'
                                    }
                                    else if (lastestPrediction == 'politeness') {
                                        lastestDetermination = 'politeness'
                                    }
                                    else {
                                        lastestDetermination = ""
                                        lastestPrediction = ""
                                    }
                                }
                                resetStatus();
                                if (lastestSentence != "" && lastestDetermination != "" && lastestPrediction != "") {
                                    $.ajax({
                                        url: 'http://127.0.0.1:5000/labeling',
                                        contentType: 'application/json;charset=UTF-8',
                                        method: 'POST',
                                        dataType: "html",
                                        data: JSON.stringify({'sentence': lastestSentence, 'label': lastestDetermination, 'pred': lastestPrediction}),
                                        success: function (response) {
                                            console.log(response);
                                            updateDisplayList(JSON.parse(response).results)
                                        }
                                    })
                                }
                            }
                            
                            function updateDisplayList(res) {
                                var obscenityList = res.obscenity
                                var politenessList = res.politeness
                                var sizeOL = obscenityList.length;
                                var sizePL = politenessList.length;
                                
                                document.getElementById("politenessList").innerHTML = '';
                                document.getElementById("obscenityList").innerHTML = '';
                                
                                document.getElementById("numPolite").innerText = 'number of polite sentences : ' + sizePL;
                                document.getElementById("numObsence").innerText = 'number of impolite sentences : ' + sizeOL;
                                
                                var s = ""
                                
                                for (var i = 0; i < sizeOL ; i++) {
                                    if (obscenityList[i][1] == 'p') {
                                        s += '<p style="word-wrap:break-word;"><span style="background-color: lightblue;">● ' + obscenityList[i][0] + '</span></p>';
                                    }
                                    else {
                                        s += '<p style="word-wrap:break-word;"><span>● ' + obscenityList[i][0] + '</span></p>';
                                    }
                                }
                                document.getElementById("obscenityList").innerHTML = s;
                                s = ""
                                for (var i = 0; i < sizePL ; i++) {
                                    if (politenessList[i][1] == 'o') {
                                        s += '<p style="word-wrap:break-word;"><span style="background-color: pink;">● ' + politenessList[i][0] + '</span></p>';
                                    }
                                    else {
                                        s += '<p style="word-wrap:break-word;"><span>● ' + politenessList[i][0] + '</span></p>';
                                    }
                                }
                                document.getElementById("politenessList").innerHTML = s;
                            }
                            
                            function resetStatus() {
                                document.getElementById("politeness").innerText = 'politeness : not determined yet';
                                document.getElementById("obscenity").innerText = 'obscenity : not determined yet';
                                document.getElementById("conclusion").innerHTML = 'conclusion : not determined yet';
                            }
                            
                            function updateResult(res) {
                                lastestPrediction = res.conclusion
                                if (lastestPrediction == 'writeFile') {
                                    lastestPrediction = ""
                                    return;
                                }
                                document.getElementById("politeness").innerText = 'politeness : ' + res.politeness + ' percent';
                                document.getElementById("obscenity").innerText = 'obscenity : ' + res.obscenity + ' percent';
                                document.getElementById("conclusion").innerText = 'conclusion : ' + ((res.conclusion == 'obscenity')? 'หยาบคาย':'สุภาพ');
                                document.getElementById("conclusion").innerHTML += '<div style="word-spacing: 50px;"><button onclick="sendLabel(true);">Correct</button> <button onclick="sendLabel(false);">Incorrect</button></div>';
                            }
                        </script>
                    </body>
                    </html>
                    '''

        @app.route("/bad")
        def bad():
            return "<h1><center> ความสุภาพ = " + str(sentence2Prop[-1][0]) + " % <br><br> ความหยาบคาย = " + str(sentence2Prop[-1][1]) + " % <br><br> Machine says: ประโยคที่ท่านส่งมาหยาบคาย! <br> <a href=' " + myWebUrl_ + "sender'>back to sender</a> </center></h1>"

        @app.route("/good")
        def good():
            return "<h1><center> ความสุภาพ = " + str(sentence2Prop[-1][0]) + " % <br><br> ความหยาบคาย = " + str(sentence2Prop[-1][1]) + " % <br><br> Machine says: ประโยคที่ท่านส่งมาสุภาพ! <br> <a href=' " + myWebUrl_ + "sender'>back to sender</a> </center></h1>"

        @app.route('/sender', methods=['GET', 'POST'])
        def sender():
            if request.method == 'GET':
                htmlAllSents = "<br>"
                for a, b in sorted(sentenceCollection.items()):
                    c = str(b)
                    if not(c == 'Bad'):
                        c = 'Good'
                    htmlAllSents += str(a) + "  :  " + c + " <br>"
                lenAllSents = str(len(sentenceCollection))

                return '''
                        <html>
                          <head>
                            <title>Home Page</title>
                          </head>
                          <body>
                            <center>
                            <h1>พิมพ์ประโยคหรือวลี</h1>
                            <form action=" ''' + myWebUrl_ + '''sender" method="post">
                                <input type="text" size="80" name="projectFilepath"><br><br>
                                <input type="checkbox" name="isBad" value="Bad"> Bad (ถ้าหยาบคายให้ติ๊กถูก) <br><br>
                                <input type="checkbox" name="writeFile" value="shouldWrite"> Write File (กดปุ่มนี้ ถ้าต้องการให้ระบบบันทึกประโยคทั้งหมดที่เห็นในหน้าเว็บลงไฟล์ txt หลังจากนั้น list ของประโยคที่แสดงในหน้าเว็บจะถูก clear) <br><br>
                                <input type="submit"> <br><br>
                                ถ้าต้องการแก้ไขค่า Good , Bad ของประโยคใดๆ ให้พิมพ์ประโยคนั้นอีกครั้งแล้ว ติ๊กๆ/ไม่ติ๊ก จากนั้น Submit มาใหม่ <br><br>
                                สิ่งที่ Write File ไปแล้วไม่สามารถแก้ไขผ่านหน้าเว็บนี้ได้
                            </form>
                            <br><br>
                            ประโยคทั้งหมดที่ได้รับมาขณะนี้มีทั้งหมด ''' + lenAllSents + ''' ประโยค
                            <br><br>
                            ''' + htmlAllSents + '''
                            </center>
                          </body>
                        </html>
                        '''
            elif request.method == 'POST':
                s = str(request.form.get("projectFilepath"))
                if len(s) == 0:
                    return redirect(myWebUrl_ + "good")

                isBad = request.form.get("isBad")
                wf = request.form.get("writeFile")

                sentenceCollection[s] = isBad

                pd = pred(s)

                sentence2Prop.append([ pd[0][0] * 100 / (pd[0][0] + pd[0][1]) , pd[0][1] * 100 / (pd[0][0] + pd[0][1]) ])

                print(pd)
                print (s)
                print (isBad)

                if wf == "shouldWrite":
                    fName = (str(datetime.now())[:10]).replace(" ", "")
                    fName = fName.replace("-", "")
                    fName = fName.replace(":", "")
                    fName = fName.replace(".", "")
                    with codecs.open(writeSentencesCollectionPath_ + "/" + fName + '.txt', 'a', 'utf-8') as out:
                        for a, b in sorted(sentenceCollection.items()):
                            c = str(b)
                            if not (c == 'Bad'):
                                c = 'Good'
                                out.write(str(a) + "  :  " + c + "\n")
                    sentenceCollection.clear()

                if (pd[0][0] > pd[0][1]):
                    return redirect(myWebUrl_ + "good")
                else:
                    return redirect(myWebUrl_ + "bad")
            else:
                return ("Not get nor post")

        app.run()

        while 1:
            print('keyboard input : ', end="")
            s = input()
            s_token = []
            for e in s:
                s_token.append(dicEncode[e])

            oh1, tg1 = vectorOfWord2OneHot([[s_token, [0, 0]]])
            pd = sess.run(prediction, {data: oh1, dropout: dropOutVal})
            print(pd)
            if (pd[0][0] > pd[0][1]):
                print('พูดจาสุภาพ')
            else:
                print('พูดจาหยาบคาย')

"""
readFile()
tokenization()
makeBatch()

print("+++++++++++++++++++++++++++++++++++++++++++++++")
for a, b in sorted(dicDecode.items()):
    print("%s --> %r" % (a, b))

print(len(trainingSet))
print(len(testSet))

print(len(trainingSet_token))
print(len(testSet_token))

for i in range(1, 100):
    print(trainingSet[i])
    print(testSet[i])
    print("==============================")
    print(trainingSet_token[i])
    print(decodeToString(trainingSet_token[i][0]))
    print(testSet_token[i])
    print(decodeToString(testSet_token[i][0]))
    print("______________________________")

print(len(batchTrain))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(len(batchTrain[0]))
print(batchTrain[0][0][0])
print(len(batchTrain[0][0]))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(len(batchTrain[1]))
print(batchTrain[1][0][0])
print(len(batchTrain[1][0]))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(len(batchTrain[2]))
print(batchTrain[2][0][0])
print(len(batchTrain[2][0]))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(len(batchTrain[3]))
print(batchTrain[3][0][0])
print(len(batchTrain[3][0]))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(len(batchTrain[4]))
print(batchTrain[4][0][0])
print(len(batchTrain[4][0]))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

print (testSet_token[0])
"""

vocab_size_ = 90
#print ("vocab_size = ",vocab_size_)
#training()
testModel()


