# ECommerceReviewAnalysis

Sentiment Analysis (Classification) on E-Commerce Clothing Reviews.
Training an evaluation using this [kaggle data set](https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews)

data_preprocessing.py
- Prepocessing of data set, e. g. stopword and special character elimination
- You can also plot the word occurences here

keras_sentiment_classification.py:
- CNN for sentiment classification of the above mentioned data set. Splits in positive (Rating >= 3)and negative (Rating < 3) Reviews.
- Word vector represantation as word embeddings using pre defined vectors
- You can choose between [GloVe](https://nlp.stanford.edu/projects/glove/) or [word2vec](https://code.google.com/archive/p/word2vec/) embeddings
- The pre-trained word embeddings are not part of this repo, due to their size (you can download it from here: [GloVe](http://nlp.stanford.edu/data/glove.6B.zip) (822MB) or [word2vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) (3.5GB))

svm_classification.py:
- Reference implementation for comparison purposes
- Uses either linear or RBF SVM from scikit-learn
- Feature extraction with bag-of-words or TF-IDF

keras_sentiment_classification-mult_filtersizes.py:
- same approach as keras_sentiment_classification.py but with multiple different filtersizes

word2vec_trainer.py
- trains word embeddings using word2vec implementation from [gensim](https://radimrehurek.com/gensim/) on the ecommerce data set