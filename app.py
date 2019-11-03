#importing libraries
from flask import jsonify, request,Flask
import torch
import torch.nn as nn ##neural network
from torch import optim #optimizers
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
import itertools
import model
import dataprocessing as dp
from dataprocessing import Voc

voc=Voc("data")

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu") # if true  GPU will be used instead of CPU

path="dataset"
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64
MAX_LENGTH=10

# Set checkpoint to load from; set to None if starting from scratch
checkpoint_iter = 4000
loadFilename = os.path.join(path,
                            '{}_checkpoint.tar'.format(checkpoint_iter))

# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    #checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = model.EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = model.LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [dp.indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, ques):
    input_sentence = ''
    try:
        global voc
        # Get input sentence
        input_sentence = ques
        # Normalize sentence
        input_sentence = dp.normalizeString(input_sentence)
        # Evaluate sentence
        output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
        return 'Bot:', ' '.join(output_words)

    except KeyError:
        return "Error: Encountered unknown word."

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = model.GreedySearchDecoder(encoder, decoder)

# Begin chatting (uncomment and run the following line to begin)
#evaluateInput(encoder, decoder, searcher, voc)


# app
app = Flask(__name__)
#app = flask.Flask(__name__, template_folder='templates')
'''
@app.route('/')
def main():
    return(flask.render_template('index.html'))
if __name__ == '__main__':
    app.run()
'''


# routes
# routes
@app.route('/', methods=['POST'])
def predict():
    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    data=data["ques"]
    # predictions
    result = evaluateInput(encoder, decoder, searcher,data)

    # send back to browser
    output = {'results': result}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run()
