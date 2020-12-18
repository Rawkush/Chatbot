import numpy
import json
from sklearn.preprocessing import OneHotEncoder
from spacy.lang.en import English
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import pickle

nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)
with open("intents.json") as file:
    raw_data = json.load(file)

#print(data)

PAD_Token=0
MAX_LENGTH=12



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
        #tmp=[ self.getIndexOfWord(wrds) for wrds in words]
        tmp=[ 0 for i in range(self.num_words)]
        for wrds in words:
            tmp[self.getIndexOfWord(wrds)]=1
        return tmp
    
 
    def getTag(self, tag):
        return self.tags[tag]
    
    def getVocabSize(self):
        return self.num_words
    
    def getTagSize(self):
        return self.num_tags

    def addResponse(self, tag, responses):
        self.response[tag]=responses
        
        
def splitDataset(data):
    x_train=[ data.getQuestionInNum(x) for x in data.questions]
    y_train=[data.getTag(data.questions[x]) for x in data.questions]
    return x_train,y_train

    
        

data=voc()

for intent in raw_data["intents"]:
    tag=intent["tag"]
    data.addTags(tag)

    for question in intent["patterns"]: 
        ques=question.lower()
        data.addQuestion(ques,tag)


x_train,y_train=splitDataset(data)
x_train=numpy.array(x_train)
y_train=numpy.array(y_train)
#normalize
#x_train=x_train/255
#reshape ytrain
y_train = y_train.reshape((len(y_train), 1))

encoder = OneHotEncoder(sparse=False)
y_train=encoder.fit_transform(y_train)



#intialising the ANN
model = models.Sequential()

# adding first layer
model.add(layers.Dense(units = 12, input_dim = len(x_train[0])))
model.add(layers.Activation('relu'))
#adding 2nd hidden layer
model.add(layers.Dense(units = 8))
model.add(layers.Activation('relu'))
#adding output layer
model.add(layers.Dense(units = 38))
model.add(layers.Activation('softmax'))

# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN model to training set
model.fit(x_train, y_train, batch_size = 10, epochs = 100)

model.save('mymodel.h5')



#removing questions from data as its not needed it will be entered by user
#we need other info to decode prediction to text so save it inpickle
data.questions={}

# save answers from json to pickle
for intent in raw_data["intents"]:
    tag=intent["tag"]
    response=[]
    for resp in intent["responses"]: 
        response.append(resp)
    data.addResponse(tag,response)
    
    
with open('mydata.pickle', 'wb') as handle:
    pickle.dump(data, handle)

 

# predecting the test set Results
x_test=numpy.array(x_train[0])
img = numpy.expand_dims(x_test, axis = 0)
y_pred = model.predict(img)
p=numpy.argmax(y_pred, axis=1)




