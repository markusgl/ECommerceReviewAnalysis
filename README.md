# ECommerceReviewAnalysis

Sentiment Analysis (Classification) on E-Commerce Clothing Reviews.
Training an evaluation using this [kaggle data set](https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews)

keras_classification.py:
- Uses word embeddings and Convolutional Neural Networks
- You can choose between pre-trained [GloVe](https://nlp.stanford.edu/projects/glove/) (400k words of English Wikipedia) or word2vec embeddings from [google](https://code.google.com/archive/p/word2vec/)
- The word embeddings are not part of this repo, due to their size (you can download it from here: [GloVe](http://nlp.stanford.edu/data/glove.6B.zip) (822MB) or [word2vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) (1.5GB))

svm_classification.py:
- Linear SVM classification with scikit learn