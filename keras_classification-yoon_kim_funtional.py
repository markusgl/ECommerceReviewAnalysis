import os
from keras import Input
from keras.engine import Model
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Activation, Conv2D, MaxPooling2D, Merge, \
    Reshape, Convolution2D
from keras.layers import Convolution1D, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.callbacks import TensorBoard
from data_preprocessing import DataPreprocessor
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import re
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from data_preprocessing import DataPreprocessor
from keras import regularizers


BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'data/glove.6B')
MAX_SEQUENCE_LENGTH = 120
VALIDATION_SPLIT = 0.2
WORD2VEC = True
KEEP_WORDS = 1500

#labels_index = {'pos': 0, 'neutral': 1, 'neg': 2}  # dictionary mapping label name to numeric id
labels_index = {'pos': 0, 'neg': 1}  # dictionary mapping label name to numeric id


data_preprocessor = DataPreprocessor()
#positive_reviews, neutral_reviews, negative_reviews, labels = data_preprocessor.separate_pos_neutral_neg()
#texts = positive_reviews + neutral_reviews + negative_reviews
positive_reviews, negative_reviews, labels = data_preprocessor.separate_pos_neg()
texts = positive_reviews + negative_reviews

tokenizer = Tokenizer(num_words=KEEP_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
max_sequence_len = 0
for sequence in sequences:
    if len(sequence) > max_sequence_len:
        max_sequence_len = len(sequence)
print("max sequence len: %i" % max_sequence_len)
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
#labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

if WORD2VEC:
    # USE WORD2VEC WORD EMBEDDINGS
    dp = DataPreprocessor()
    # use self trained word2vec embeddings based one the same data set
    #EMBEDDING_DIM = 100
    #embeddings_index = dp.get_embeddings_index('models/w2vmodel.bin')

    # use pretrained word2vec embeddings from google
    EMBEDDING_DIM = 300
    embeddings_index = dp.get_embeddings_index_from_google_model()
else:
    # USE PRETRAINED GLOVE WORD EMBEDDINGS (trained on 20 newsgroups)
    EMBEDDING_DIM = 100
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

print(embeddings_index)

# get the vector representation of the words
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

convs = []
filter_sizes = [3,4,5]

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

for fsz in filter_sizes:
    l_conv = Conv1D(nb_filter=128,filter_length=fsz,activation='relu')(embedded_sequences)
    l_pool = MaxPooling1D(5)(l_conv)
    convs.append(l_pool)

l_merge = Merge(mode='concat', concat_axis=1)(convs)
l_cov1 = Conv1D(128, 5, activation='relu')(l_merge)
l_pool1 = MaxPooling1D(5)(l_cov1)
l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
l_pool2 = MaxPooling1D(30)(l_cov2)
l_flat = Flatten()(l_pool2)
l_dense = Dense(128, activation='relu')(l_flat)
preds = Dense(2, activation='softmax')(l_dense)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - more complex convolutional neural network")
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=2, batch_size=16, verbose=1)