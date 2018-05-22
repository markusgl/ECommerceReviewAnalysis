import os
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Activation, Merge, Dropout
from keras.layers.embeddings import Embedding
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from data_preprocessing import DataPreprocessor
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
import pickle
from keras import callbacks


BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'data/glove.6B')
MAX_SEQUENCE_LENGTH = 1000
VALIDATION_SPLIT = 0.2
WORD2VEC = True
KEEP_WORDS = 2000

#labels_index = {'pos': 0, 'neutral': 1, 'neg': 2}  # dictionary mapping label name to numeric id
labels_index = {'pos': 0, 'neg': 1}  # dictionary mapping label name to numeric id

data_preprocessor = DataPreprocessor()
#positive_reviews, neutral_reviews, negative_reviews, labels = data_preprocessor.separate_pos_neutral_neg()
#texts = positive_reviews + neutral_reviews + negative_reviews
texts, labels = data_preprocessor.separate_pos_neg()

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

data = pad_sequences(sequences, maxlen=max_sequence_len)
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


# get the vector representation of the words and create weight matrix
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, index in tokenizer.word_index.items():
    if index > KEEP_WORDS -1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[index] = embedding_vector

print(embedding_matrix)

# set parameters:
BATCH_SIZE = 16
FILTERS = 300
KERNEL_SIZES = (3, 4, 5)
EPOCHS = 3
HIDDEN_DIMS = 250
P_DROPOUT = 0.5

submodels = []
for kernel_size in KERNEL_SIZES:
    submodel = Sequential()
    submodel.add(Embedding(len(word_index) + 1,
                           EMBEDDING_DIM,
                           weights=[embedding_matrix],
                           input_length=max_sequence_len,
                           trainable=False))
    submodel.add(Dropout(P_DROPOUT))
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
#model.add(Dense(1))
model.add(Dense(len(labels_index)))
model.add(Activation('sigmoid'))
#model.add(Activation('softmax'))

print('Compiling model')
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Log to tensorboard
tensorBoardCallback = TensorBoard(log_dir='./logs/sequential_mult_filters', write_graph=True)
# Callbacks
checkpointer = ModelCheckpoint(filepath='models/sentiment_sequential.hdf5', verbose=1, save_best_only=True)
earlyStopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)


model.fit([x_train, x_train, x_train],
          y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=([x_val, x_val, x_val], y_val),
          callbacks=[checkpointer, earlyStopper, reduce_lr],
          verbose=2)
model.summary()


# Evaluation on the test set
scores = model.evaluate(x_val, y_val, verbose=0, batch_size=BATCH_SIZE)
print("Accuracy: %.2f%%" % (scores[1]*100))
print("Loss: %.2f%%" % (scores[0]*100))
# TODO confusion matrix

