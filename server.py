from flask import Flask,redirect
from flask import request, jsonify, json, make_response, current_app
from flask_cors import CORS, cross_origin
from collections import defaultdict

app = Flask(__name__)
app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={r"/test": {"origins": "http://localhost:5000"}})

myList = [set(),set()]

@app.route('/test', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content-Type','Authorization'])
def check_rudeness():
    data = request.get_json()
    if 'sentence' in data:
        print (data['sentence'])
        machineSays = {'conclusion': 'obscenity', 'obscenity': '50', 'politeness': '50'}

        return jsonify(results=machineSays)

    noInputSentence = {'Error' : 'cant find key name "sentence"'}
    return jsonify(results=noInputSentence)

@app.route('/labeling', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content-Type','Authorization'])
def labeling():
    data = request.get_json()
    if 'sentence' in data:
        if 'label' in data:
            if data['label'] == 'politeness':
                myList[0].add(data['sentence'])
            elif data['label'] == 'obscenity':
                myList[1].add(data['sentence'])
    print ("*-----------------------------------------------*")
    print (myList[0])
    print (myList[1])
    return jsonify(results= {'politeness': list(myList[0]), 'obscenity': list(myList[1])})

@app.route('/sender')
def sender():
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
                            <button onclick="sendRequest();">Determine the value of politeness/obscenity</button>
                        </h1>
                    </div>
                    <br>
                    <div>
                        <p id="politeness">politeness : not determined yet</p>
                        <p id="obscenity">obscenity : not determined yet</p>
                        <p id="conclusion">conclusion : not determined yet</p>		
                    </div>
                </center>
                <br>
                <div style="width: 80%; margin: auto;">
                    <div style="float:left; width: 50%; background-color: lightblue;">
                        <p style="word-wrap:break-word;">● aadasdasdadaasdasdasdasddda</p>
                        <p style="word-wrap:break-word;">● 5555555555555555555555555555</p>
                    </div>
                    
                    <div style="float:right; width: 50%; background-color: pink;">
                        <p style="word-wrap:break-word;">● bbgfgfgasdadasddfgfgfgb</p>
                        <p style="word-wrap:break-word;">● 6666666666666666666666666</p>
                    </div>			
                </div>
                <script>
                    function sendRequest() {
                        $.ajax({
                            url: 'https://38b28dc5.ngrok.io/test',
                            contentType: 'application/json;charset=UTF-8',
                            method: 'POST',
                            dataType: "html",
                            data: JSON.stringify({'sentence': document.getElementById("sentence").value}),
                            success: function (response) {
                                console.log(response);
                                testAlert(JSON.parse(response).results);
                            }
                        })
                    }
                    
                    function testAlert(res) {
                        document.getElementById("politeness").innerText = 'politeness : ' + res.politeness + ' percent'
                        document.getElementById("obscenity").innerText = 'obscenity : ' + res.obscenity + ' percent'
                        document.getElementById("conclusion").innerText = 'conclusion : ' + ((res.conclusion == 'obscenity')? 'Your sentence/phrase is impolite.':'Your sentence/phrase is polite.')
                        document.getElementById("conclusion").innerHTML += '<div style="word-spacing: 50px;"><button>Correct</button> <button>Incorrect</button></div>'
                    }
                </script>
            </body>
            </html>
            '''

app.run()
