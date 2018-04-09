import pandas as pd
import re
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from nltk.tokenize import WhitespaceTokenizer
from sklearn.decomposition import PCA
import numpy as np

keep_words_and_punct = r"[^a-zA-Z?!.]|[.]{2,}"
keep_words = r"[^a-zA-Z]|[.]{2,}"
mult_whitespaces = "\s{3,}"

df = pd.read_csv('data/review_data.csv')

positive_reviews = []
negative_reviews = []
for i, row in df.iterrows():
    clean_review = re.sub(mult_whitespaces, ' ', re.sub(keep_words_and_punct, ' ', str(row['Review Text']).lower()))
    if row['Rating'] >= 3:
        positive_reviews.append(clean_review)
    else:
        negative_reviews.append(clean_review)

# split sentences and tokenize each sentence to a list
train_sentences = []
tokenizer = WhitespaceTokenizer()
for review in negative_reviews:
    sentences = re.split("[.?!]", str(review))
    for sentence in sentences:
        train_sentences.append(tokenizer.tokenize(sentence))

print("start training...")
model = Word2Vec(train_sentences, size=100, window=5, min_count=5, workers=4)
model.save('data/w2vmodel.bin')

X = model[model.wv.vocab]
vocab_size = len(model.wv.vocab)
vector_dim = len(model.wv['the'])

# store embeddings to numpy matrix for tf and keras
def get_embedding_matrix():
    model = Word2Vec.load('data/w2vmodel.bin')
    embedding_matrix = np.zeros((len(model.wv.vocab), vector_dim))
    for word in range(len(model.wv.vocab)):
        embedding_vector = model.wv[model.wv.index2word[word]]
        if embedding_vector is not None:
            embedding_matrix[word] = embedding_vector

    return embedding_matrix

# Visualization using matplotlib and PCA
"""
pca = PCA(n_components=2)
result = pca.fit_transform(X)

plt.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.savefig('wordvectors.png')
plt.show()
"""