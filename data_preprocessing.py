import pandas as pd
import re
from gensim.models import Word2Vec, KeyedVectors
import matplotlib.pyplot as plt
from nltk.tokenize import WhitespaceTokenizer
from sklearn.decomposition import PCA
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import operator
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle

class DataPreprocessor:

    def separate_pos_neg(self):
        keep_words_and_punct = r"[^a-zA-Z0-9?!.]|[\.]{2,}"
        keep_words = r"[^a-zA-Z0-9]|[\.]{2,}"
        mult_whitespaces = "\s{2,}"

        df = pd.read_csv('data/review_data.csv')
        df.dropna(how="any", inplace=True)  # drop blank lines
        # df = pd.read_csv('data/review_data_small.csv')

        texts = []
        labels = []
        stop_words = set(stopwords.words('english'))

        #file = open('data/duplicate_words', 'rb')
        #duplicate_words = pickle.load(file)
        #file.close()
        duplicate_words = ['dress', 'size', 'top', 'fit', 'like']

        for i, row in df.iterrows():
            review = str(row['Title']) + '. ' + str(row['Review Text'])
            #clean_review = re.sub(mult_whitespaces, ' ', re.sub(keep_words_and_punct, ' ', str(review).lower()))
            clean_review = re.sub(mult_whitespaces, ' ', re.sub(keep_words, ' ', str(review).lower()))
            tokens = word_tokenize(clean_review)
            #filtered_sentence = [word for word in tokens if not word in stop_words and not word in duplicate_words]
            filtered_sentence = [word for word in tokens if not word in stop_words]
            sentences = " ".join(filtered_sentence)
            #print(sentences)

            if row['Rating'] >= 3:
                texts.append(sentences)
                labels.append(0)
            else:
                texts.append(sentences)
                labels.append(1)

        return texts, labels

    def separate_pos_neutral_neg(self):
        keep_words_and_punct = r"[^a-zA-Z?!.]|[.]{2,}"
        keep_words = r"[^a-zA-Z]|[.]{2,}"
        mult_whitespaces = "\s{3,}"

        df = pd.read_csv('data/review_data.csv')
        df.dropna(how="any", inplace=True)  # drop blank lines

        # df = pd.read_csv('data/review_data_small.csv')
        positive_reviews = []
        negative_reviews = []
        neutral_reviews = []
        labels = []
        for i, row in df.iterrows():
            review = str(row['Title']) + '. ' + str(row['Review Text'])
            clean_review = re.sub(mult_whitespaces, ' ', re.sub(keep_words_and_punct, ' ', str(review).lower()))
            if row['Rating'] > 3:
                positive_reviews.append(clean_review)
                labels.append(0)
            elif row['Rating'] == 3:
                neutral_reviews.append(clean_review)
                labels.append(1)
            else:
                negative_reviews.append(clean_review)
                labels.append(2)

        return positive_reviews, neutral_reviews, negative_reviews, labels

    def clean_and_separate_reviews(self):
        keep_words_and_punct = r"[^a-zA-Z?!.]|[.]{2,}"
        keep_words = r"[^a-zA-Z]|[.]{2,}"
        mult_whitespaces = "\s{3,}"
        df = pd.read_csv('data/review_data.csv')

        clean_reviews = []
        for i, row in df.iterrows():
            review = str(row['Title']) + '. ' + str(row['Review Text'])
            clean_review = re.sub(mult_whitespaces, ' ', re.sub(keep_words_and_punct, ' ', str(review).lower()))
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

    def train_word2vec(self):
        train_sentences = self.split_and_tokenize_reviews()
        # start word embeddings training
        print("start training word2vec...")
        model = Word2Vec(train_sentences, size=100, window=5, min_count=5, workers=4)
        print("training completed")
        print("vocabulary length %i" % len(model.wv.vocab))
        model.save('models/w2vmodel.bin')


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
        model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True, limit=15000)

        embeddings_index = {}
        for word in range(len(model.wv.vocab)):
            embedding_vector = model.wv[model.wv.index2word[word]]
            embeddings_index[model.wv.index2word[word]] = embedding_vector

        return embeddings_index

    # Visualization using matplotlib and PCA
    def plot_model(self):
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
        df = pd.read_csv('data/review_data.csv')
        count_pos = 0
        count_neg = 0
        count_neu = 0

        for i, row in df.iterrows():
            if row['Rating'] >= 3:
                count_pos += 1
            #elif row['Rating'] == 3:
            #    count_neu += 1
            else:
                count_neg += 1

        print("Positive Reviews: %i" % count_pos)
        print("Negative Reviews: %i" % count_neg)
        print("Neutral Reviews: %i" % count_neu)

    def count_reviews_length(self):
        df = pd.read_csv('data/review_data.csv')
        max_len = 0

        for i, row in df.iterrows():
            words = (str(row['Title']) + '. ' + str(row['Review Text'])).split()
            print(words)
            review_length = len(Counter(words))
            if review_length > max_len:
                max_len = review_length

        print("max review length: %i" % max_len)

    def count_word_occurences(self):
        text, labels = DataPreprocessor().separate_pos_neg()
        #print(len(labels))
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

        amount_unique_pos_tokens = len(pos_word_index.keys())
        amount_unique_neg_tokens = len(neg_word_index.keys())

        sorted_pos = sorted(pos_word_index.items(), key=operator.itemgetter(1), reverse=True)
        sorted_neg = sorted(neg_word_index.items(), key=operator.itemgetter(1), reverse=True)
        print(sorted_pos)
        print(sorted_neg)

        #print(len(pos_token_list))
        sorted_pos_short = {}
        sorted_neg_short = {}
        for i in range(10):
            tuple_pos = sorted_pos[i]
            sorted_pos_short[tuple_pos[0]] = (tuple_pos[1] / len(pos_token_list))*100
            tupple_neg = sorted_neg[i]
            sorted_neg_short[tupple_neg[0]] = (tupple_neg[1] / len(neg_token_list))*100


        """
        duplicate_list = []
        for token in pos_token_list:
            if token in neg_token_list:
                duplicate_list.append(token)
        """
        #file = open('data/duplicate_words', 'wb')
        #pickle.dump(duplicate_list, file)
        #file.close()
        #print(duplicate_list)

        """
        plt.bar(range(len(sorted_pos_short)), list(sorted_pos_short.values()), align='center', color='b')
        plt.xlabel('Wörter')
        plt.ylabel('Vorkommen in Prozent')
        plt.title('Zehn häufigsten Wörter in Positivliste')
        plt.xticks(range(len(sorted_pos_short)), list(sorted_pos_short.keys()))
        plt.savefig('wordcount_pos_wo_clean.png')
        plt.show()

        plt.bar(range(len(sorted_neg_short)), list(sorted_neg_short.values()), align='center', color='r')
        plt.xlabel('Wörter')
        plt.ylabel('Vorkommen in Prozent')
        plt.title('Zehn häufigsten Wörter in Negativliste')
        plt.xticks(range(len(sorted_neg_short)), list(sorted_neg_short.keys()))
        plt.savefig('wordcount_neg_wo_clean.png')
        plt.show()

        """

        duplicate_pos_short = {}
        duplicate_neg_short = {}
        for key in sorted_pos_short.keys():
            if key in sorted_neg_short.keys():
                duplicate_pos_short[key] = sorted_pos_short[key]
                duplicate_neg_short[key] = sorted_neg_short[key]
        #print(duplicate_pos_short)
        #print(duplicate_neg_short)

        X = np.arange(len(duplicate_neg_short.keys()))
        pos_bar = plt.bar(X + 0.10, list(duplicate_pos_short.values()), width=0.30, color='b')
        neg_bar = plt.bar(X + 0.40, list(duplicate_neg_short.values()), width=0.30, color='r')
        plt.xlabel('Wörter')
        plt.ylabel('Vorkommen in Prozent')
        plt.title('Zehn häufigsten Wörter in beiden Listen')
        #plt.set_xticks(X + 0.25 / 2)
        plt.xticks(X + 0.25 / 2, list(duplicate_neg_short.keys()))
        plt.legend((pos_bar[0], neg_bar[0]), ('Positiv', 'Negativ'), loc='upper right')
        plt.savefig('wordcount_pos_neg_wo_clean.png', format='png')
        plt.show()


#DataPreprocessor().count_word_occurences()
