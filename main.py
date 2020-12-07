
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import time
import difflib
import numpy
import webbrowser
import tflearn
import tensorflow
import random
from flask import Flask, render_template, request
import json
import pickle
import os

app = Flask(__name__)
with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels =[]
    docs_patt = []
    docs_tag = []


    for intent in data["intents"]:
        # below we fetch patterns from all intents in one place
        for pattern in intent["patterns"]:
            # below we put each word from pattern in the list wrds and then append it to words list and append the pattern in docs
            wrds = nltk.word_tokenize(pattern)
            for item in wrds:
                words.extend(wrds)
                docs_patt.append(wrds)
                docs_tag.append(intent["tag"])
                # here we add all labels in the list of labels
                if intent["tag"] not in labels:
                    labels.append(intent["tag"])

        # here we take each ord from words list and then find its root word
    words = [stemmer.stem(w.lower()) for w in words]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_patt):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_tag[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)




# this fuction creates list of words from the sentence
def words_to_list(s):
    a = []
    ns = ""
    s = s + " " 
    for i in range(len(s)):
        if s[i] == " ":
            a.append(ns)
            ns = ""
        else:
            ns = ns + s[i]
    a = list(set(a))
    return a

# pass the file in this fuction to create a dictionary of unique vocabulary
def json_to_dictionary(data):
    dictionary = []
    fil_dict= []
    vocalubary = []
    for i in data["intents"]:
        for pattern in i["patterns"]:
            vocalubary.append(pattern.lower())
    for i in vocalubary:
        dictionary.append(words_to_list(i))
    for i in range(len(dictionary)):
        for word in dictionary[i]:
            fil_dict.append(word)
    return list(set(fil_dict))

# this fuction checks the spelling in the sentence
chatbot_vocabulary = json_to_dictionary(data)

def word_checker(s):
    correct_string = ""
    for word in s.casefold().split():
        if word not in chatbot_vocabulary:
            suggestion = difflib.get_close_matches(word, chatbot_vocabulary)
            for x in suggestion:
                pass
            if len(suggestion) == 0:
                pass
            else:
                correct_string = correct_string + " " + str(suggestion[0])
        else:
            correct_string = correct_string + " " + str(word)

    return correct_string 


def chat(inp):
    while True:
        if inp.lower() == "bye":
            return "Good Bye!"
            
        
        inp_x = word_checker(inp)
        print(inp_x)

        results = model.predict([bag_of_words(inp_x, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        print(results[results_index])

        if inp == "":
            return "Hey ask me some questions like...courses available at GEC?  "
        elif results[results_index] >= 1.0:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            return random.choice(responses)

        else:
            return "Sorry, I don't know how to answer that yet "

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
        """ 
   print("*"*60)
    time.sleep(1)
    print("*"*40)
    time.sleep(1)
    print("*"*20)
    time.sleep(1)
    print("ChatBot will be launched in your web browser in..")
    time.sleep(1)
    print("3 sec")
    time.sleep(1)
    print("2 sec")
    time.sleep(1)
    print("1 sec")
    time.sleep(1)
    print("0 sec")
    webbrowser.open("http://127.0.0.1:5000/") 
    """
