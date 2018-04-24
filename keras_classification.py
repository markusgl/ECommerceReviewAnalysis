import os
from keras import Input
from keras.engine import Model
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D
from keras.layers import Convolution1D, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.callbacks import TensorBoard
from data_preprocessing import DataPreprocessor
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import re
from keras.utils import to_categorical


BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'data/glove.6B')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

texts = []
labels = []
labels_index = {'pos': 0, 'neg': 1}  # dictionary mapping label name to numeric id

keep_words_and_punct = r"[^a-zA-Z?!.]|[.]{2,}"
keep_words = r"[^a-zA-Z]|[.]{2,}"
mult_whitespaces = "\s{3,}"

df = pd.read_csv('data/review_data.csv')
# df = pd.read_csv('data/review_data_small.csv')
positive_reviews = []
negative_reviews = []
for i, row in df.iterrows():
    clean_review = re.sub(mult_whitespaces, ' ', re.sub(keep_words_and_punct, ' ', str(row['Review Text']).lower()))
    if row['Rating'] >= 3:
        positive_reviews.append(clean_review)
        labels.append(0)
    else:
        negative_reviews.append(clean_review)
        labels.append(1)

texts = positive_reviews + negative_reviews
print(labels_index)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

print(labels)
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT  * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


# PREPARING EMBEDDING LAYER
# TODO use another word embedding set and word2vec
# use self trained embeddings
"""
dp = DataPreprocessor()
embedding_matrix = dp.get_embedding_matrix('data/w2vmodel.bin')
"""

# USE PRETRAINED WORD EMBEDDINGS
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# KERAS TRAINING
from keras.layers import Embedding

print("EMBEDDING_DIM %s" % embedding_matrix.shape[1])
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print("start training...")
tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=3, callbacks=[tensorBoardCallback], batch_size=128, verbose=2)

# Evaluation on the test set
scores = model.evaluate(x_val, y_val, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

