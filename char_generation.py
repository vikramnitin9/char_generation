from keras.models import Sequential
from keras.layers.core import Dense, Activation, RepeatVector
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
from keras.utils import np_utils
from keras.layers import Dropout

from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

from six.moves import cPickle

import numpy as np
import sys

fileptr = open("data/input.txt","r")
raw_text = fileptr.read()

chars = sorted(list(set(raw_text)))
print(chars)

char_to_index = dict((char,i) for i,char in enumerate(chars))
index_to_char = dict((i,char) for i,char in enumerate(chars))

max_len = 50
num_chars = len(chars)
sample_size = len(raw_text)

print("Length of training data is ", sample_size)
print("Number of distinct characters are ", num_chars)

x_train = []
y_train = []

for i in range(0, sample_size - max_len):
	x_train.append(raw_text[i : i+max_len]) # Will give an array of strings
	y_train.append(raw_text[i+max_len])

X_train = np.zeros((len(x_train), max_len, num_chars), dtype=np.bool)
Y_train = np.zeros((len(y_train), num_chars), dtype=np.bool)

for i in range(0, len(x_train)):
	for j,char in enumerate(x_train[i]):
		X_train[i, j, char_to_index[char]] = 1
	Y_train[i, char_to_index[y_train[i]]] = 1

model = Sequential()
model.add(LSTM(512, input_shape=(max_len, num_chars), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(num_chars))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer='RMSprop')

# model.fit(X_train, Y_train, batch_size=100, nb_epoch=5)

# model.save_weights('gandhi1.h5')
model.load_weights('gandhi1.h5')

# helper function to sample an index from a probability array
# Higher temperature -> More diversity but more mistakes
# Temp ~ 0 -> most likely output. Highly repetitive

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# For predicting, pick a random starting value and take max_len chars from there
# That forms the seed sequence

gen_chars = 1000 # Number of chars to be generated

seed_start = np.random.randint(0, len(X_train)-1)
seed = x_train[seed_start]

print("---------------------Seed used is : ---------------------")
print(seed)

output_sequence = ""

for i in range(0, gen_chars):
	inp = np.zeros((1,max_len,num_chars))
	for j,char in enumerate(seed):
		inp[0,j,char_to_index[char]] = 1

	pred = model.predict(inp)[0]
	next_index = sample(pred, 0.5)
	next_char = index_to_char[next_index]

	output_sequence += next_char
	seed += next_char
	seed = seed[1:] # 1 to 101, 2 to 102 and so on...

print("--------------------The generated sequence is : ----------------------")
print(output_sequence)