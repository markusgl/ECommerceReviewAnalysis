from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

import itertools
import os
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Activation, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD

from data_preprocessing import DataPreprocessor
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'data/glove.6B')
# MAX_SEQUENCE_LENGTH = 150
# MAX_NUM_WORDS = 3000
VALIDATION_SPLIT = 0.2
WORD2VEC = True

labels_index = {'pos': 0, 'neg': 1}
data_preprocessor = DataPreprocessor()
texts, labels = data_preprocessor.separate_pos_neg()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print(word_index)
print('Found %s unique tokens.' % len(word_index))
max_sequence_len = 0
for sequence in sequences:
    if len(sequence) > max_sequence_len:
        max_sequence_len = len(sequence)
print("max sequence len: %i" % max_sequence_len)

data = pad_sequences(sequences, maxlen=max_sequence_len)
labels = to_categorical(np.asarray(labels))
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
    data_preprocessor = DataPreprocessor()

    # use self trained word2vec embeddings based one the same data set
    # EMBEDDING_DIM = 100
    # embeddings_index = data_preprocessor.get_embeddings_index('data/w2vmodel.bin')

    # use pretrained word2vec embeddings from google
    EMBEDDING_DIM = 300
    embeddings_index = data_preprocessor.get_embeddings_index_from_google_model()
else:
    # USE PRETRAINED GLOVE WORD EMBEDDINGS (trained on 20 newsgroups)
    EMBEDDING_DIM = 100

    if os.path.isfile('data/embeddings_index.pkl'):
        with open('data/embeddings_index.pkl', 'rb') as file:
            embeddings_index = pickle.load(file)
    else:
        embeddings_index = {}
        f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        with open('data/embeddings_index.pkl', 'wb') as file:
            pickle.dump(embeddings_index, file)

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


def create_model(activation, optimizer='adam', dropout_rate='0.5'):

    # set parameters:
    FILTERS = 250
    KERNEL_SIZE = 3
    HIDDEN_DIMS = 250
   # P_DROPOUT = 0.5

    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                        EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=max_sequence_len,
                        trainable=False))

    model.add(Dropout(dropout_rate))
    #model.add(BatchNormalization())
    model.add(Conv1D(FILTERS,
                     KERNEL_SIZE,
                     padding='valid',
                     activation='relu',
                     strides=1))

    model.add(GlobalMaxPooling1D())
    model.add(Dense(HIDDEN_DIMS))
    model.add(Dropout(dropout_rate))
    model.add(Activation('relu'))
    model.add(Dense(len(labels_index)))
    model.add(Activation(activation))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model

model = KerasClassifier(build_fn=create_model)

#hyperparams
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
epochs = [10,20,30]
batch_size = [16, 32, 64]
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

param_grid = dict(activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(x_train, y_train, verbose=2)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))