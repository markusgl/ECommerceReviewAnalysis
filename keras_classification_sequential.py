import itertools
import os
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Activation, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD, Adamax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_preprocessing import DataPreprocessor
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,  accuracy_score, \
    f1_score, precision_score, recall_score

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'data/glove.6B')
#MAX_SEQUENCE_LENGTH = 150
#MAX_NUM_WORDS = 3000
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
    #EMBEDDING_DIM = 100
    #embeddings_index = data_preprocessor.get_embeddings_index('data/w2vmodel.bin')

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

# set parameters:
BATCH_SIZE = 32
FILTERS = 300
KERNEL_SIZE = 3
HIDDEN_DIMS = 250
EPOCHS = 50
P_DROPOUT = 0.2

model = Sequential()
model.add(Embedding(len(word_index) + 1,
                    EMBEDDING_DIM,
                    weights=[embedding_matrix],
                    input_length=max_sequence_len,
                    trainable=False))  # prevent keras from updating the word indices during training process

model.add(Dropout(P_DROPOUT))
#model.add(BatchNormalization())
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
#model.add(Activation('softmax'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Callbacks
tensorBoardCallback = TensorBoard(log_dir='./logs/sequential', write_graph=True)
checkpointer = ModelCheckpoint(filepath='models/sentiment_sequential.hdf5', verbose=1, save_best_only=True)
earlyStopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=(x_val, y_val),
          callbacks=[checkpointer, earlyStopper, reduce_lr],
          verbose=2)


# Evaluation on the test set
scores = model.evaluate(x_val, y_val, verbose=0, batch_size=BATCH_SIZE)
print("Accuracy: %.2f%%" % (scores[1]*100))
print("Loss: %.2f%%" % (scores[0]*100))


########### CROSS VALIDATION ############
"""
# scores
ac_scores = []
f1_scores = []
prec_scores = []
rec_scores = []

confusion = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]])

kfold = StratifiedKFold(n_splits=10, random_state=1).split(texts, labels)

for k, (train_indices, test_indices) in enumerate(kfold):
    #model.fit(x_train[train_indices], y_train[train_indices])

    train_text = texts[train_indices]
    train_y = labels[train_indices]

    test_text = texts[test_indices]
    test_y = labels[test_indices]

    model.fit(train_text, train_y)
    predictions = model.predict(test_text)

    confusion += confusion_matrix(test_y, predictions)

    ac_scores.append(accuracy_score(test_y, predictions))
    f1_scores.append(f1_score(test_y, predictions))
    prec_scores.append(precision_score(test_y, predictions))
    rec_scores.append(recall_score(test_y, predictions))

print("---------------------- \nResults for ", 'CNN', " with ", "word embeddings" ":")
print("K-Folds Accuracy-score: ", sum(ac_scores) / len(ac_scores))
print("K-Folds F1-score: ", sum(f1_scores) / len(f1_scores))
print("K-Folds Precision-score: ", sum(prec_scores) / len(prec_scores))
print("K-Folds Recall-score: ", sum(rec_scores) / len(rec_scores))

print("CV accuracy : %.3f +/- %.3f" % (np.mean(ac_scores), np.std(ac_scores)))
"""

