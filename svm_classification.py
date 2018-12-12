import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.pipeline import Pipeline
import time
from data_preprocessing import DataPreprocessor

# Data preprocessing
data_preprocessor = DataPreprocessor()
texts, labels = data_preprocessor.separate_pos_neg()

# split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=1)

pipeline = Pipeline([
                    ('vect', TfidfVectorizer(ngram_range=(1, 2), max_df=0.5, use_idf=False, sublinear_tf=True)),
                    #('vect', CountVectorizer(ngram_range=(1, 2), max_df=0.5)),
                    ('clf', SVC(kernel='rbf', C=100, gamma=0.01, decision_function_shape='ovo', probability=True))
                    ])

def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

print("Start training SVM...")
start = time.time()
pipeline.fit(X_train, y_train)
print("traing time %s" % str(time.time()-start))
print("Training finished")
print("Starting validation...")
scores = pipeline.score(X_test, y_test)
print(scores)