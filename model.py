import numpy
import json

with open("intents.json") as file:
    data = json.load(file)

#print(data)

class voc:
    def __init__(self):
        self.num_words= 0  # number of the words we have
        self.tags=[]
        self.questions={}
        self.words2index={}
        self.index2words={}

    def addWord(self,word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.index2words[self.num_words]=word
            self.num_words += 1

    def addQuestion(self, question, answer):
        self.questions[question]=answer
    
    def addTags(self,tag):
        self.tags.append(tag)
    


for intent in data["intents"]:
    for pattern in intent["patterns"]:
        