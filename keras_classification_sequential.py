import os
from keras import Input
from keras.engine import Model
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Activation
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

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'data/glove.6B')
MAX_SEQUENCE_LENGTH = 1000
#MAX_NUM_WORDS = 20000
VALIDATION_SPLIT = 0.2
WORD2VEC = True

texts = []
labels = []
labels_index = {'pos': 0, 'neg': 1}  # dictionary mapping label name to numeric id

keep_words_and_punct = r"[^a-zA-Z?!.]|[.]{2,}"
keep_words = r"[^a-zA-Z']|[.]{2,}"
mult_whitespaces = "\s{2,}"

df = pd.read_csv('data/review_data.csv')
# df = pd.read_csv('data/review_data_small.csv')
df.dropna(how="any", inplace=True)

# split in to positive and negative reviews
positive_reviews = []
negative_reviews = []
for i, row in df.iterrows():
    #clean_review = re.sub(mult_whitespaces, ' ', re.sub(keep_words_and_punct, ' ', str(row['Review Text']).lower()))
    clean_review = re.sub(mult_whitespaces, ' ', re.sub(keep_words, ' ', str(row['Review Text']).lower()))
    #print(clean_review)
    if row['Rating'] >= 3:
        positive_reviews.append(clean_review)
        labels.append(0)
    else:
        negative_reviews.append(clean_review)
        labels.append(1)

texts = positive_reviews + negative_reviews
#print(labels_index)

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
x_test = data[-nb_validation_samples:]
y_test = labels[-nb_validation_samples:]

if WORD2VEC:
    # USE WORD2VEC WORD EMBEDDINGS
    EMBEDDING_DIM = 300
    dp = DataPreprocessor()
    # use self trained word2vec embeddings based one the same data set
    #embeddings_index = dp.get_embeddings_index('data/w2vmodel.bin')
    # use pretrained word2vec embeddings from google
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

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# set parameters:
#max_features = 5000
MAX_SEQUENCE_LEN = 1000
BATCH_SIZE = 16
#embedding_dims = 50
FILTERS = 250
KERNEL_SIZE = 3
HIDDEN_DIMS = 250
EPOCHS = 3


model = Sequential()
# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(len(word_index) + 1,
                    EMBEDDING_DIM,
                    weights=[embedding_matrix],
                    input_length=MAX_SEQUENCE_LEN))

model.add(Dropout(0.5))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(FILTERS,
                 KERNEL_SIZE,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())
# We add a vanilla hidden layer:
model.add(Dense(HIDDEN_DIMS))
model.add(Dropout(0.5))
model.add(Activation('relu'))
# We project onto a single unit output layer, and squash it with a sigmoid:
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
          validation_data=(x_test, y_test),
          callbacks=[tensorBoardCallback],
          verbose=2)

# Evaluation on the test set
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
# TODO confusion matrix
