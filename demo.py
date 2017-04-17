#goal: create word vectors from a game of thrones dataset
#and analyze them to see semantic similarity

from __future__ import absolute_import, print_function

#for word encoding
import codecs

#regex
import glob

#concurrency
import multiprocessing

#logging
import logging

#dealing with operating system,
import os

#pretty printing
import pprint

#regular exp
import re

#natural language toolkit
import nltk

#word 2 vec
import gensim.models.word2vec as w2v

#from gensim import models

#dimensionality reduction
import sklearn.manifold

#math
import numpy as np

#plotting
import matplotlib.pyplot as plt

#parse pandas
import pandas as pd

#visulization
import seaborn as sns

#setup logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#step 1 to process/clean data
nltk.download('punkt') #pretrained tokenizer
nltk.download('stopwords') 


#get the book name, matching txt file
book_filenames = sorted(glob.glob("data/*.txt"))
book_filenames


corpus_raw = u''

for book_filename in book_filenames:
	print("Reading'{0}'...".format(book_filename))
	with codecs.open(book_filename, 'r', 'utf-8') as book_file:
		corpus_raw += book_file.read()
	print("Corpus is now {0} characters long".format(len(corpus_raw)))
	print()

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(corpus_raw)

#convert into list of words
#remove unnecesary, split into words, no hyphens
#list of words
def sentence_to_wordlist(raw):
	#clean = re.sub("[^a-zA-Z]"," ",raw)
	clean = re.sub("[^a-zA-Z]"," ", raw)
	words = clean.split()
	return words
 
#sentence where each words in tokenized
sentences = []
for raw_sentence in raw_sentences:
	if len(raw_sentence) > 0:
		sentences.append(sentence_to_wordlist(raw_sentence))

print(raw_sentences[5])
print(sentence_to_wordlist(raw_sentences[5]))


token_count = sum([len(sentence) for sentence in sentences])
print("The book corpus contain {0:,} tokens".format(token_count)) 


#once we have vectors
#3 main tasks that vectors help with
#distance, similarity, ranking

#train Word2Vec

#dimensionality of the resulting word vectors
#more dimensions, more computationally to train, more accurate
num_features = 300

#minimum word count threshold
min_word_count = 3

#number of threads to run in parallel
num_workers = multiprocessing.cpu_count()

#context window length
context_size = 7

#downloading setting for frequent words
#0 - le-5 is good for this
downsampling = 1e-3

#seed for the RNG, to make the results reproducible
#random number generator
#deterministic, good for debugging
seed = 1

thrones2vec = w2v.Word2Vec(
		sg = 1,
		seed = seed,
		workers = num_workers,
		size = num_features,
		min_count = min_word_count,
		window = context_size,
		sample = downsampling
)


thrones2vec.build_vocab(sentences)

print("Word2Vec vocabulary lenght:", len(thrones2vec.wv.vocab))

thrones2vec.train(sentences)

if not os.path.exists("trained"):
	os.makedirs("trained")

thrones2vec.save(os.path.join("trained", "thrones2vec.w2v"))

#explore the trained model

thrones2vec = w2v.Word2Vec.load_word2vec_format(os.path.join("trained","thrones2vec.w2v"), binary=False)

#compress the word vectors into 2D space and plot them
tsne = sklearn.manifold.TSNE(n_components = 2, random_state = 0)
all_word_vectors_matrix = thrones2vec.syn0

#train t-SNE, this could take a minute or two
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

#plot the big picture
points = pd.DataFrame(
	[
		(words, coords[0], coords[1])
		for word, coords in [
			(word, all_word_vectors_matrix_2d[thrones2vec.wv.vocab[word].index])
			for word in thrones2vec.wv.vocab
		]
	],
	columns = ['word', 'x', 'y']
)

points.head(10)

sns.set_context('poster')
points.plot.scatter('x', 'y', s=10, figsize=(20,12))


def plot_region(x_bounds, y_bounds):
	slice = points[
		(x_bounds[0] <= points.x) &
		(points.x <= x_bounds[1]) &
		(y_bounds[0] <= points.y) &
		(points.y <= y_bounds[1])
	]
	
	ax = slice.plot.scatter('x', 'y', s=35, figsize=(10, 8))
	for i, point in slice.iterrows():
		ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize = 11)

plot_region(x_bounds = (4.0, 4.2), y_bounds = (-0.5, -0.1))

plot_region(x_bounds = (0, 1), y_bounds = (4, 4.5))

#explore semantic similarities between book characters

#Word closest to the given word
thrones2vec.most_similar('Stark')
thrones2vec.most_similar('Aerys')
thrones2vec.most_similar('direwolf')

#Linear relationships between word pairs
def nearest_similarity_cosmul(start1, end1, end2):
	similarities = thrones2vec.most_similar_cosmul(
		positive = [end2, start2],
		negative = [end1]
	)
	start2 = similarities[0][0]
	print('{start1} is related to {end1}, as {start2} is related to {end2}')
	return start2

nearest_similarity_cosmul('Stark', 'Winterfell', 'Riverrun')
nearest_similarity_cosmul('Jaime', 'sword', 'wine')
nearest_similarity_cosmul('Arya', 'Nymeria', 'dragons')






