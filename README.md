# ECommerceReviewAnalysis

Sentiment Analysis (Classification) on E-Commerce Clothing Reviews.
Training an evaluation using this [kaggle data set](https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews)

keras_classification.py:
- Using pre-trained [GloVe](https://nlp.stanford.edu/projects/glove/) word embeddings (400k words of English Wikipedia) and Convolutional Neural networks
- The word embeddings are not part of this repo, due to their size (822MB you can downlaod it from [here](http://nlp.stanford.edu/data/glove.6B.zip))

svm_classification.py:
- Linear SVM classification with scikit learn and tfidf