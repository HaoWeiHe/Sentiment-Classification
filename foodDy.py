import collections
import os
from flask import Flask
# from ner import pattern_append
# flask_ngrok_example.py
from flask import Flask
from flask import Flask, render_template, request
# from flask_ngrok import run_with_ngrok

import spacy
spacy.load('en_core_web_sm')
import en_core_web_sm
nlp = en_core_web_sm.load()

from spacy import displacy

def displayNLP(example):
  return displacy.render(example, jupyter=False, style='ent').replace("#ddd","#E0777D")

from flask import Flask
from flask_mail import Mail, Message
app = Flask(__name__)


app.config.update(dict(
    MAIL_SERVER = 'smtp.googlemail.com',
    MAIL_PORT = 465,
    MAIL_USE_TLS = False,
    MAIL_USE_SSL = True,
    MAIL_USERNAME = 'e899876412@gmail.com',
    MAIL_PASSWORD = 'Demo#12345'

))
app.debug = True
mail = Mail(app)

# run_with_ngrok(app)
CATGs = {'OIL':6, 'CERAL':3, 'DIARY':1.5, 'FRUIT':3, 'MEAT':6, 'VEG':4}


import json
@app.route("/sentMail", methods=[ 'POST']) 
def sentMail():
    contactName = request.values['contactName']
    contactSubject = request.values['contactSubject']
    contactMessage = request.values['contactMessage']
    recipients = request.values['contactEmail']
  

    error = {}
    if (len(contactName) < 2):
        error['Name Error'] = "Please enter your name."
    
    # // Check Email
    import re

    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if (not re.match(regex, recipients)):
        error['Email Error'] = "Please enter a valid email address."
    
    # // Check Message
    if (len(contactMessage) < 15) :
        error['Message Error'] = "Please enter your message. It should have at least 15 characters.";
    
    # // Subject
    if not contactSubject:
        contactSubject = "Contact Form Submission"; 

    if error:
        return json.dumps(error)

    msg = Message(contactSubject, sender='e899876412@gmail.com', recipients=["altar396@gmail.com"])
    

    msg.body = "contactSubject:\t{}\n from:\t{}\ncontactMessage:\t{}\n ".format(contactSubject, recipients,contactMessage)#'This is a test email' #Customize based on user input
    mail.send(msg)
    return "OK"

from knowledgeGraph import get_relation, get_entity  


@app.route("/KGapi", methods=['POST'])
def KGapi():
    from markupsafe import Markup
    text = request.values['KG_text']
    rel = get_relation(text)
    ents = get_entity(text)
    nlp_text = nlp(text)
    dep = displacy.render(nlp_text, jupyter=False, style='dep')
    dep = Markup(dep)

    from combinehtml import GenerateHtml
    ner = Markup(GenerateHtml().getTable(ents[0], ents[1], rel))
    cy = Markup(GenerateHtml().getCy(ents[0], ents[1], rel))
    
    return json.dumps({"KG_text":text, "depViz":dep, "nerViz":ner,"cy":cy}), 200, {'ContentType':'application/json'} 


@app.route("/SentimentSubmit", methods=['GET'])
def SentimentSubmit():
    text = request.values['user_text']
    import SentimentClassifier
    res = SentimentClassifier.predict(text) 
    pred_probs = SentimentClassifier.pred_probs(text)[int(res)] * 100
    print(res)
    level = 100
    if res == 1:
        return render_template('SentimentSubmit.html', **locals() )
    return render_template('SentimentSubmit_neg.html', **locals())


@app.route("/multiLabelSubmit", methods=['GET'])
def multiLabelSubmit():
    
    text = request.values['user_text']
    from violation import predict
    result = predict(text, get_probs = 1)
    score = sum(predict(text).values())
    result =  {k:v*100 for k,v in result.items()  }
    # prmint(score)
    result["text"] = text
    if score < 1:
        return render_template('multiLabelSubmitValid.html',**result )
    
    return render_template('multiLabelSubmit.html',**result )



@app.route("/submit", methods=['POST'])
def submit():
    from markupsafe import Markup
    corpus = request.values['hiddeninput']
    ents = []
    deps = []
    examples = []
    for stn in corpus.split("#"):
      if not stn:
        continue
      example = getTextDoc(stn)
      ent = displayNLP(example)
      ents.append(Markup(ent))
      dep = displayDep(example)
      deps.append(Markup(dep ))
      examples.append(example)
    len_ent = len(ents)
    len_dep = len(deps)
    counter = collections.defaultdict(int)
    for example in examples:
      for ent in example.ents:
        if ent.label_ in CATGs:
          counter[ent.label_] += 1
      
    fat, Carbohydrate, Protein,VEGnFruit = 100 * float(counter["OIL"] + counter["DIARY"])/ (CATGs["OIL"] + CATGs["DIARY"]), \
                                            100 * float(counter['CERAL'])/ CATGs["CERAL"], \
                                            100 * float(counter['MEAT'])/ CATGs["MEAT"], \
                                            100 * float(counter['FRUIT'] + counter['VEG'])/ (CATGs["VEG"]+CATGs["FRUIT"])
    return render_template('submit1.html',**locals() )

@app.route("/KGPage")
def KGPage():
    return render_template('KGPage.html')



@app.route("/sentimentalPage")
def sentimentalPage():
    return render_template('sentimentalPage.html')


@app.route("/test")
def test():
    return render_template('blog-single.html')


@app.route("/multiLabelDemo")
def multiLabelDemo():
    return render_template('multiLabelDemo.html')

@app.route("/sentimentDemo")
def sentimentDemo():
    return render_template('sentimentDemo.html')



@app.route("/multiLabelPage")
def multiLabelPage():
    return render_template('multiLabelPage.html')


@app.route("/form")
def form():
    return render_template('form.html')

@app.route("/hello")
def hello():
    from markupsafe import Markup
    res = displayNLP("I ate an apple today.")
    NER = Markup(res)
    return render_template('index.html', **locals())

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/tripletViz1")
def tripletViz():
    return render_template("tripletViz1.html")

@app.route("/KGDemo")
def KGDemo():
    return render_template("KGDemo.html")

@app.route("/speechRecognize", methods=['POST', 'GET'])
def speechRecognize():
      return render_template('speechRecognization.html')#, request="POST")

 
if __name__ == '__main__':
    # app.run(host = "https://h0u1eud9p9h-496ff2e9c6d22116-64-colab.googleusercontent.com/")
    # app.debug = True
    # app.run()
  
    # mail = Mail(app)

    # from werkzeug.contrib.fixers import ProxyFix
    # app.wsgi_app = ProxyFix(app.wsgi_app)
    
    app.run()
    # app.run(host='0.0.0.0')

