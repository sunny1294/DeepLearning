{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import nltk\n",
    "import numpy as np\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from nltk.stem import WordNetLemmatizer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from bs4 import BeautifulSoup\n",
    "\n",
    "\n",
    "wordnet_lemmatizer = WordNetLemmatizer()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "stopwords = set(w.rstrip() for w in open('stopwords.txt'))\n",
    "\n",
    "positive_reviews = BeautifulSoup(open('electronics/positive.review').read())\n",
    "positive_reviews = positive_reviews.findAll('review_text')\n",
    "\n",
    "negative_reviews = BeautifulSoup(open('electronics/negative.review').read())\n",
    "negative_reviews = negative_reviews.findAll('review_text')\n",
    "\n",
    "np.random.shuffle(positive_reviews)\n",
    "positive_reviews = positive_reviews[:len(negative_reviews)]\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def my_tokenizer(s):\n",
    "    s = s.lower() # downcase\n",
    "    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)\n",
    "    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful\n",
    "    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form\n",
    "    tokens = [t for t in tokens if t not in stopwords] # remove stopwords\n",
    "    return tokens\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "word_index_map = {}\n",
    "current_index = 0\n",
    "positive_tokenized = []\n",
    "negative_tokenized = []\n",
    "\n",
    "for review in positive_reviews:\n",
    "    tokens = my_tokenizer(review.text)\n",
    "    positive_tokenized.append(tokens)\n",
    "    for token in tokens:\n",
    "        if token not in word_index_map:\n",
    "            word_index_map[token] = current_index\n",
    "            current_index += 1\n",
    "\n",
    "for review in negative_reviews:\n",
    "    tokens = my_tokenizer(review.text)\n",
    "    negative_tokenized.append(tokens)\n",
    "    for token in tokens:\n",
    "        if token not in word_index_map:\n",
    "            word_index_map[token] = current_index\n",
    "            current_index += 1\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def tokens_to_vector(tokens, label):\n",
    "    x = np.zeros(len(word_index_map) + 1) \n",
    "    for t in tokens:\n",
    "        i = word_index_map[t]\n",
    "        x[i] += 1\n",
    "    x = x / x.sum()\n",
    "    x[-1] = label\n",
    "    return x\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "N = len(positive_tokenized) + len(negative_tokenized)\n",
    "\n",
    "data = np.zeros((N, len(word_index_map) + 1))\n",
    "i = 0\n",
    "for tokens in positive_tokenized:\n",
    "    xy = tokens_to_vector(tokens, 1)\n",
    "    data[i,:] = xy\n",
    "    i += 1\n",
    "\n",
    "for tokens in negative_tokenized:\n",
    "    xy = tokens_to_vector(tokens, 0)\n",
    "    data[i,:] = xy\n",
    "    i += 1\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "np.random.shuffle(data)\n",
    "\n",
    "X = data[:,:-1]\n",
    "Y = data[:,-1]\n",
    "\n",
    "Xtrain = X[:-100,]\n",
    "Ytrain = Y[:-100,]\n",
    "Xtest = X[-100:,]\n",
    "Ytest = Y[-100:,]\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "model = LogisticRegression()\n",
    "model.fit(Xtrain, Ytrain)\n",
    "print(\"Classification rate:\", model.score(Xtest, Ytest))\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
