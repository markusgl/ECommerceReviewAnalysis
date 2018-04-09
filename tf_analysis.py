import tensorflow as tf
import numpy as np
import data_preprocessing as dp
from gensim.models import Word2Vec

# select some random word for testing purposes and to evaluate similarity
valid_size = 16
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# get trained word2vec embeddings and put it into tensorflow
saved_embeddings = tf.constant(dp.get_embedding_matrix())
# setting trainable to False for a fixed embedding layer
embedding = tf.Variable(initial_value=saved_embeddings, trainable=False)

# create the cosine similarity operations
norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
normalized_embeddings = embedding / norm
valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

# Add variable initializer.
init = tf.global_variables_initializer()

model = Word2Vec.load('data/w2vmodel.bin')
wv = model.wv

with tf.Session() as sess:
    sess.run(init)
    sim = similarity.eval()

    for i in range(valid_size):
        valid_word = wv.index2word[i]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in range(top_k):
            close_word = wv.index2word[nearest[k]]
            log_str = '%s %s,' % (log_str, close_word)
        print(log_str)


# TODO train on positive and negative examples