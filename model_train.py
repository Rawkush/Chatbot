import numpy
import json
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import pickle
from voc import voc  

def splitDataset(data):
    x_train=[ data.getQuestionInNum(x) for x in data.questions]
    y_train=[data.getTag(data.questions[x]) for x in data.questions]
    return x_train,y_train
with open("intents.json") as file:
    raw_data = json.load(file)

  
        

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
'''
y_train = y_train.reshape((len(y_train), 1))

encoder = OneHotEncoder(sparse=False)
y_train=encoder.fit_transform(y_train)
'''


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




