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


BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'data/glove.6B')
MAX_SEQUENCE_LENGTH = 120
VALIDATION_SPLIT = 0.2
WORD2VEC = True
KEEP_WORDS = 1500

#texts = []
#labels = []
#labels_index = {'pos': 0, 'neutral': 1, 'neg': 2}  # dictionary mapping label name to numeric id
labels_index = {'pos': 0, 'neg': 1}  # dictionary mapping label name to numeric id

keep_words_and_punct = r"[^a-zA-Z?!.]|[.]{2,}"
keep_words = r"[^a-zA-Z']|[.]{2,}"
mult_whitespaces = "\s{2,}"

#df = pd.read_csv('data/review_data_tiny.csv')
# df = pd.read_csv('data/review_data_small.csv')
#df.dropna(how="any", inplace=True)

# split in to positive and negative reviews
"""
positive_reviews = []
negative_reviews = []
for i, row in df.iterrows():
    #clean_review = re.sub(mult_whitespaces, ' ', re.sub(keep_words_and_punct, ' ', str(row['Review Text']).lower()))
    review = row['Title'] + ' ' + row['Review Text']
    clean_review = re.sub(mult_whitespaces, ' ', re.sub(keep_words, ' ', str(review).lower()))
    print(clean_review)
    if row['Rating'] >= 3:
        positive_reviews.append(clean_review)
        labels.append(0)
    else:
        negative_reviews.append(clean_review)
        labels.append(1)
"""

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
x_test = data[-nb_validation_samples:]
y_test = labels[-nb_validation_samples:]

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


# set parameters:
BATCH_SIZE = 16
FILTERS = 100
KERNEL_SIZES = (3, 4, 5)
EPOCHS = 2
HIDDEN_DIMS = 250
P_DROPOUT = 0.25

submodels = []
for kernel_size in KERNEL_SIZES:    # kernel sizes
    submodel = Sequential()
    submodel.add(Embedding(len(word_index) + 1,
                           EMBEDDING_DIM,
                           weights=[embedding_matrix],
                           input_length=MAX_SEQUENCE_LENGTH,
                           trainable=False))

    submodel.add(Conv1D(filters=FILTERS,
                        kernel_size=kernel_size,
                        padding='valid',
                        activation='relu',
                        strides=1))
    submodel.add(GlobalMaxPooling1D())
    submodels.append(submodel)


model = Sequential()
model.add(Merge(submodels, mode="concat"))
model.add(Dense(HIDDEN_DIMS))
model.add(Dropout(P_DROPOUT))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
#model.add(Activation('softmax'))
print('Compiling model')
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Log to tensorboard
tensorBoardCallback = TensorBoard(log_dir='./logs/sequential_kim', write_graph=True)

model.fit([x_train, x_train, x_train],
          y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=([x_test, x_test, x_test], y_test),
          verbose = 1)

# Evaluation on the test set
#scores = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
#print("Accuracy: %.2f%%" % (scores[1]*100))
# TODO confusion matrix

