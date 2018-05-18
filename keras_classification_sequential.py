import numpy
import os
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.callbacks import TensorBoard
from sklearn.model_selection import StratifiedKFold

from data_preprocessing import DataPreprocessor
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'data/glove.6B')
MAX_SEQUENCE_LENGTH = 150
MAX_NUM_WORDS = 1500
VALIDATION_SPLIT = 0.2
WORD2VEC = False

labels_index = {'pos': 0, 'neg': 1}
data_preprocessor = DataPreprocessor()
texts, labels = data_preprocessor.separate_pos_neg()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
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
    EMBEDDING_DIM = 300
    data_preprocessor = DataPreprocessor()
    # use self trained word2vec embeddings based one the same data set
    #embeddings_index = dp.get_embeddings_index('data/w2vmodel.bin')
    # use pretrained word2vec embeddings from google
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


# set parameters:
BATCH_SIZE = 16
FILTERS = 250
KERNEL_SIZE = 3
HIDDEN_DIMS = 250
EPOCHS = 3
P_DROPOUT = 0.5

model = Sequential()
model.add(Embedding(len(word_index) + 1,
                    EMBEDDING_DIM,
                    weights=[embedding_matrix],
                    input_length=MAX_SEQUENCE_LENGTH))

model.add(Dropout(P_DROPOUT))
model.add(Conv1D(FILTERS,
                 KERNEL_SIZE,
                 padding='valid',
                 activation='relu',
                 strides=1))

model.add(GlobalMaxPooling1D())
model.add(Dense(HIDDEN_DIMS))
model.add(Dropout(P_DROPOUT))
model.add(Activation('relu'))

#model.add(Dense(1))
model.add(Dense(len(labels_index)))
model.add(Activation('sigmoid'))

# Log to tensorboard
tensorBoardCallback = TensorBoard(log_dir='./logs/sequential', write_graph=True)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=(x_val, y_val),
          callbacks=[tensorBoardCallback],
          verbose=2)


# Evaluation on the test set
scores = model.evaluate(x_val, y_val, verbose=0, batch_size=BATCH_SIZE)
print("Accuracy: %.2f%%" % (scores[1]*100))
print("Loss: %.2f%%" % (scores[0]*100))
# TODO confusion matrix

