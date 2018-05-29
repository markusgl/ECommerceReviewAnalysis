import pandas as pd
import re
from gensim.models import Word2Vec, KeyedVectors

from nltk.tokenize import WhitespaceTokenizer
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import operator
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class DataPreprocessor:

    def separate_pos_neg(self):
        """
        Preprocessing steps for ML
        Extracts the three columns Title, Review Text and Rating from the dataset
        Performs stopword removal and elimination of special characters
        """
        keep_words_and_punct = r"[^a-zA-Z0-9?!.]|[\.]{2,}"
        #keep_words = r"[^a-zA-Z0-9]|[\.]{2,}"
        mult_whitespaces = "\s{2,}"

        df = pd.read_csv('data/review_data.csv')
        reviews = df[['Title', 'Review Text', 'Rating']]
        reviews.dropna(how="any", inplace=True, subset=['Review Text', 'Rating'])

        texts = []
        labels = []
        stop_words = set(stopwords.words('english'))
        duplicate_words = ['dress', 'size', 'top', 'fit', 'like']

        for i, row in reviews.iterrows():
            review = str(row['Title']) + '. ' + str(row['Review Text'])
            clean_review = re.sub(mult_whitespaces, ' ', re.sub(keep_words_and_punct, ' ', str(review).lower()))
            #clean_review = re.sub(mult_whitespaces, ' ', re.sub(keep_words, ' ', str(review).lower()))
            tokens = word_tokenize(clean_review)
            filtered_sentence = [word for word in tokens if not word in stop_words and not word in duplicate_words]
            #filtered_sentence = [word for word in tokens if not word in stop_words]
            sentences = " ".join(filtered_sentence)

            if row['Rating'] >= 3:
                texts.append(sentences)
                labels.append(0)
            else:
                texts.append(sentences)
                labels.append(1)

        return texts, labels

    def get_embeddings_index(self, model_path):
        model = Word2Vec.load(model_path)

        embeddings_index = {}
        for word in range(len(model.wv.vocab)):
            embedding_vector = model.wv[model.wv.index2word[word]]
            embeddings_index[model.wv.index2word[word]] = embedding_vector

        return embeddings_index

    def get_embeddings_index_from_google_model(self):
        #model = KeyedVectors.load_word2vec_format('C:/develop/data/GoogleNews-vectors-negative300.bin', binary=True, limit=15000)
        model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True, limit=25000)

        embeddings_index = {}
        for word in range(len(model.wv.vocab)):
            embedding_vector = model.wv[model.wv.index2word[word]]
            embeddings_index[model.wv.index2word[word]] = embedding_vector

        return embeddings_index


    def plot_model(self):
        """
        Visualization of word2vec model using matplotlib and PCA
        """
        model = Word2Vec.load('models/w2vmodel.bin')
        #model = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
        #model = KeyedVectors.load_word2vec_format('models/w2vmodel.bin')

        X = model[model.wv.vocab]
        pca = PCA(n_components=2)
        result = pca.fit_transform(X)

        plt.scatter(result[:, 0], result[:, 1])
        words = list(model.wv.vocab)
        print(words)
        for i, word in enumerate(words):
            plt.annotate(word, xy=(result[i, 0], result[i, 1]))

        plt.savefig('wordvectors.png')
        plt.show()

    def count_reviews(self):
        """
        simply counts the entries in each class for analysis purposes
        :return:
        """
        df = pd.read_csv('data/review_data.csv')
        count_pos = 0
        count_neg = 0
        reviews = df[['Title', 'Review Text', 'Rating']]
        reviews.dropna(how="any", inplace=True, subset=['Review Text', 'Rating'])

        for i, row in reviews.iterrows():
            if row['Rating'] >= 3:
                count_pos += 1
            else:
                count_neg += 1

        print("Positive Reviews: %i" % count_pos)
        print("Negative Reviews: %i" % count_neg)

    def count_word_occurences(self, save=False):
        text, labels = DataPreprocessor().separate_pos_neg()

        pos_list = []
        neg_list = []
        for sequence in text:
            if labels[text.index(sequence)] == 0:
                pos_list.append(sequence)
            else:
                neg_list.append(sequence)

        pos_token_list = []
        pos_word_index = {}
        for sequence in pos_list:
            tokens = WhitespaceTokenizer().tokenize(sequence)

            for token in tokens:
                pos_token_list.append(token)

                if token in pos_word_index.keys():
                    pos_word_index[token] += 1
                else:
                    pos_word_index[token] = 1

        neg_token_list = []
        neg_word_index = {}
        for sequence in neg_list:
            tokens = WhitespaceTokenizer().tokenize(sequence)
            for token in tokens:
                neg_token_list.append(token)

                if token in neg_word_index:
                    neg_word_index[token] += 1
                else:
                    neg_word_index[token] = 1

        sorted_pos = sorted(pos_word_index.items(), key=operator.itemgetter(1), reverse=True)
        sorted_neg = sorted(neg_word_index.items(), key=operator.itemgetter(1), reverse=True)

        sorted_pos_short = {}
        sorted_neg_short = {}
        for i in range(10):
            tuple_pos = sorted_pos[i]
            sorted_pos_short[tuple_pos[0]] = (tuple_pos[1] / len(pos_token_list))*100
            tupple_neg = sorted_neg[i]
            sorted_neg_short[tupple_neg[0]] = (tupple_neg[1] / len(neg_token_list))*100


        # Plot most frequent words in positive class before stop word elimination
        plt.bar(range(len(sorted_pos_short)), list(sorted_pos_short.values()), align='center', color='b')
        plt.xlabel('Wörter')
        plt.ylabel('Vorkommen in Prozent')
        plt.title('Zehn häufigsten Wörter in Positivliste')
        plt.xticks(range(len(sorted_pos_short)), list(sorted_pos_short.keys()))
        if save:
            plt.savefig('wordcount_pos_wo_clean.png')
        plt.show()

        # Plot most frequent words in negative class before stop word elimination
        plt.bar(range(len(sorted_neg_short)), list(sorted_neg_short.values()), align='center', color='r')
        plt.xlabel('Wörter')
        plt.ylabel('Vorkommen in Prozent')
        plt.title('Zehn häufigsten Wörter in Negativliste')
        plt.xticks(range(len(sorted_neg_short)), list(sorted_neg_short.keys()))
        if save:
            plt.savefig('wordcount_neg_wo_clean.png')
        plt.show()

        # Plot most frequent words in both class before stop word elimination
        duplicate_pos_short = {}
        duplicate_neg_short = {}
        for key in sorted_pos_short.keys():
            if key in sorted_neg_short.keys():
                duplicate_pos_short[key] = sorted_pos_short[key]
                duplicate_neg_short[key] = sorted_neg_short[key]

        X = np.arange(len(duplicate_neg_short.keys()))
        pos_bar = plt.bar(X + 0.10, list(duplicate_pos_short.values()), width=0.30, color='b')
        neg_bar = plt.bar(X + 0.40, list(duplicate_neg_short.values()), width=0.30, color='r')
        plt.xlabel('Wörter')
        plt.ylabel('Vorkommen in Prozent')
        plt.title('Zehn häufigsten Wörter in beiden Klassen')
        plt.xticks(X + 0.25 / 2, list(duplicate_neg_short.keys()))
        plt.ylim([0,7])
        plt.legend((pos_bar[0], neg_bar[0]), ('Positiv', 'Negativ'), loc='upper right')
        if save:
            plt.savefig('wordcount_pos_neg_wo_clean.png', format='png')
        plt.show()



