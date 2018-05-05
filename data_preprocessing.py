import pandas as pd
import re
from gensim.models import Word2Vec, KeyedVectors
import matplotlib.pyplot as plt
from nltk.tokenize import WhitespaceTokenizer
from sklearn.decomposition import PCA
import numpy as np


class DataPreprocessor:

    def separate_pos_neg(self):
        keep_words_and_punct = r"[^a-zA-Z?!.]|[.]{2,}"
        keep_words = r"[^a-zA-Z]|[.]{2,}"
        mult_whitespaces = "\s{3,}"

        df = pd.read_csv('data/review_data.csv')
        df.dropna(how="all", inplace=True)  # drop blank lines
        # df = pd.read_csv('data/review_data_small.csv')
        positive_reviews = []
        negative_reviews = []
        for i, row in df.iterrows():
            review = row['Title'] + ' ' + row['Review Text']
            print(review)
            clean_review = re.sub(mult_whitespaces, ' ', re.sub(keep_words_and_punct, ' ', str(review).lower()))
            print(clean_review)
            if row['Rating'] >= 3:
                positive_reviews.append(clean_review)
            else:
                negative_reviews.append(clean_review)

        return positive_reviews, negative_reviews

    def separate_pos_neutral_neg(self):
        keep_words_and_punct = r"[^a-zA-Z?!.]|[.]{2,}"
        keep_words = r"[^a-zA-Z]|[.]{2,}"
        mult_whitespaces = "\s{3,}"

        df = pd.read_csv('data/review_data.csv')
        df.dropna(how="all", inplace=True)  # drop blank lines
        # df = pd.read_csv('data/review_data_small.csv')
        positive_reviews = []
        negative_reviews = []
        neutral_reviews = []
        for i, row in df.iterrows():
            clean_review = re.sub(mult_whitespaces, ' ', re.sub(keep_words_and_punct, ' ', str(row['Review Text']).lower()))
            if row['Rating'] > 3:
                positive_reviews.append(clean_review)
            elif row['Rating'] == 3:
                neutral_reviews.append(clean_review)
            else:
                negative_reviews.append(clean_review)

        return positive_reviews, negative_reviews

    def clean_and_separate_reviews(self):
        keep_words_and_punct = r"[^a-zA-Z?!.]|[.]{2,}"
        keep_words = r"[^a-zA-Z]|[.]{2,}"
        mult_whitespaces = "\s{3,}"
        df = pd.read_csv('data/review_data.csv')

        clean_reviews = []
        for i, row in df.iterrows():
            clean_review = re.sub(mult_whitespaces, ' ', re.sub(keep_words_and_punct, ' ', str(row['Review Text']).lower()))
            clean_reviews.append(clean_review)
        print("clean_reviews_length %s"%len(clean_reviews))
        return clean_reviews

    def split_and_tokenize_reviews(self):
        # split sentences and tokenize each sentence to a list
        reviews = self.clean_and_separate_reviews()
        train_sentences = []
        tokenizer = WhitespaceTokenizer()
        for review in reviews:
            sentences = re.split("[.?!]", str(review))
            for sentence in sentences:
                train_sentences.append(tokenizer.tokenize(sentence))
        print("train_sentences length %s"%len(train_sentences))
        return train_sentences

    def train_word2vec(self, pos_net_split=False):
        # TODO implement pos_neg_split
        train_sentences = self.split_and_tokenize_reviews()
        # start word embeddings training
        print("start training word2vec...")
        model = Word2Vec(train_sentences, size=100, window=5, min_count=5, workers=4)
        print("training completed")
        print(len(model.wv.vocab))
        model.save('data/w2vmodel.bin')


    # store embeddings to numpy matrix for tensorflow and keras
    def get_embedding_matrix(self, model_path):
        model = Word2Vec.load(model_path)
        # vocab_size = len(model.wv.vocab)
        # get the dimension of the word vectors
        vector_dim = len(model.wv['the']) # TODO more robust solution

        # create appropriate numpy zeros array
        embedding_matrix = np.zeros((len(model.wv.vocab), vector_dim))

        # iterate over embedding representation of each word in the vocabulary and add it to the np array
        for word in range(len(model.wv.vocab)):
            embedding_vector = model.wv[model.wv.index2word[word]]
            if embedding_vector is not None:
                embedding_matrix[word] = embedding_vector

        return embedding_matrix

    def get_embeddings_index(self, model_path):
        model = Word2Vec.load(model_path)

        embeddings_index = {}
        for word in range(len(model.wv.vocab)):
            embedding_vector = model.wv[model.wv.index2word[word]]
            embeddings_index[model.wv.index2word[word]] = embedding_vector

        return embeddings_index

    def get_embeddings_index_from_google_model(self):
        model = KeyedVectors.load_word2vec_format('C:/develop/Data/GoogleNews-vectors-negative300.bin', binary=True)

        embeddings_index = {}
        for word in range(len(model.wv.vocab)):
            embedding_vector = model.wv[model.wv.index2word[word]]
            embeddings_index[model.wv.index2word[word]] = embedding_vector

        return embeddings_index

    # Visualization using matplotlib and PCA
    def plot_model(self):
        #model = Word2Vec.load(model_path)
        model = KeyedVectors.load_word2vec_format('C:/Users/marku/develop/Data/GoogleNews-vectors-negative300.bin',
                                                  binary=True)
        X = model[model.wv.vocab]
        pca = PCA(n_components=2)
        result = pca.fit_transform(X)

        plt.scatter(result[:, 0], result[:, 1])
        words = list(model.wv.vocab)
        for i, word in enumerate(words):
            plt.annotate(word, xy=(result[i, 0], result[i, 1]))

        plt.savefig('wordvectors.png')
        plt.show()


#DataPreprocessor().train_word2vec()
#DataPreprocessor().get_embedding_matrix('data/w2vmodel.bin')
#DataPreprocessor().plot_model()
