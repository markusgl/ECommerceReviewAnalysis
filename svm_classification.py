import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.pipeline import Pipeline

keep_words_and_punct = r"[^a-zA-Z?!.]|[.]{2,}"
keep_words = r"[^a-zA-Z]|[.]{2,}"
mult_whitespaces = "\s{3,}"

df = pd.read_csv('../data/review_data.csv')
df.dropna(how="all", inplace=True) # drop blank lines

review_text = re.sub(mult_whitespaces, ' ', re.sub(keep_words_and_punct, ' ', str(df['Review Text']).lower()))
ratings = df['Rating'].values # ratings from 0 to 5

# split ratings into only positive (0) and negative (1) reviews
targets = np.zeros(shape=ratings.shape, dtype=int)
for i in range(len(ratings)):
    if ratings[i] >= 3:
        targets[i, ] = 0
    else:
        targets[i, ] = 1

pipeline = Pipeline([
                    ('vect', TfidfVectorizer(ngram_range=(1, 2), max_df=0.5, use_idf=False, sublinear_tf=True)),
                    ('clf', SVC(kernel='linear', C=100, gamma=0.01, decision_function_shape='ovo', probability=True))
                    ])

print("Start training SVM...")
#pipeline.fit(word_counts, targets)
pipeline.fit(df['Review Text'].values.astype(str), targets)
print("Training finished")
print("Starting cross validation...")
scores = cross_val_score(pipeline, df['Review Text'].values.astype(str), targets, cv=5)
#scores = cross_val_score(pipeline, word_counts, targets, cv=5)
print(scores)