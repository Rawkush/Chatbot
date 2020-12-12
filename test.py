
from sklearn.preprocessing import OneHotEncoder
from spacy.lang.en import English
import numpy
from flask import Flask, render_template, request
import json
import pickle
import os
import time
import keras
from keras.models import Sequential
from keras.models import load_model

nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)
PAD_Token=0
MAX_LENGTH=12

app = Flask(__name__)
     
model= load_model('mymodel.h5')

class voc:
    
    def __init__(self):
        self.num_words= 1  # 0 is reserved for padding 
        self.num_tags=0
        self.tags={}
        self.index2tags={}
        self.questions={}
        self.word2index={}
        self.response={}
  
    def addWord(self,word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.num_words += 1

    def addTags(self,tag):
        if tag not in self.tags:
            self.tags[tag]=self.num_tags
            self.index2tags[self.num_tags]=tag
            self.num_tags+=1

    def addQuestion(self, question, answer):
        self.questions[question]=answer
        words=self.tokenization(question)
        for  wrd in words:
            self.addWord(wrd)
            
        
    def tokenization(self,ques):
        tokens = tokenizer(ques)
        token_list = []
        for token in tokens:
            token_list.append(token.lemma_)
        return token_list
    
    def getIndexOfWord(self,word):
        return self.word2index[word]
    
    def getQuestionInNum(self, ques):
        words=self.tokenization(ques)
        tmp=[ self.getIndexOfWord(wrds) for wrds in words]
        while(len(tmp)<MAX_LENGTH):
            tmp.append(PAD_Token)
        return tmp
    
    def getTag(self, tag):
        return self.tags[tag]
    
    def getVocabSize(self):
        return self.num_words
    
    def getTagSize(self):
        return self.num_tags

    def addResponse(self, tag, responses):
        self.response[tag]=responses
       
with open("mydata.pickle", "rb") as f:
    data = pickle.load(f)



def predict(ques):
    ques= data.getQuestionInNum(ques)
    ques=numpy.array(ques)
    ques=ques/255
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
 
