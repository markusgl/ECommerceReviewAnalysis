import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

keep_words_and_punct = r"[^a-zA-Z?!.]|[.]{2,}"
keep_words = r"[^a-zA-Z]|[.]{2,}"
mult_whitespaces = "\s{3,}"

df = pd.read_csv('../data/review_data.csv')
df.dropna(how="all", inplace=True) # drop blank lines

review_text = re.sub(mult_whitespaces, ' ', re.sub(keep_words_and_punct, ' ', str(df['Review Text']).lower()))
targets = df['Rating'].values

vectorizer = TfidfVectorizer()
word_counts = vectorizer.fit_transform(df['Review Text'].values.astype(str)).astype(float)

clf = SVC(kernel='linear', C=100, gamma=0.01, decision_function_shape='ovo', probability=True)
print("Start training SVM...")
clf.fit(word_counts, targets)
print("Training finished")
scores = cross_val_score(clf, word_counts, targets, cv=5)
print(scores)