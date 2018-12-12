""" Trains a word2vec model from dataset """
import re
import pandas as pd
from nltk.tokenize import WhitespaceTokenizer
from gensim.models import Word2Vec


class Word2VecTrainer:
    def clean_and_separate_reviews(self):
        keep_words_and_punct = r"[^a-zA-Z?!.]|[.]{2,}"
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
