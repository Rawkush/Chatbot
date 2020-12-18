from spacy.lang.en import English
import numpy
from flask import Flask, render_template, request
import json
import pickle
import os
import time
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from voc import voc

nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)
PAD_Token=0

app = Flask(__name__)
     
model= models.load_model('mymodel.h5')
        
with open("mydata.pickle", "rb") as f:
    data = pickle.load(f)


def predict(ques):
    ques= data.getQuestionInNum(ques)
    ques=numpy.array(ques)
   # ques=ques/255
    ques = numpy.expand_dims(ques, axis = 0)
    y_pred = model.predict(ques)
    res=numpy.argmax(y_pred, axis=1)
    return res
    

def getresponse(results):
    tag= data.index2tags[int(results)]
    response= data.response[tag]
    return response

def chat(inp):
    while True:
        inp_x=inp.lower()
        results = predict(inp_x)
        response= getresponse(results)
        return response[0]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    time.sleep(1)
    return str(chat(userText))

if __name__ == "__main__":
        app.run()
 
