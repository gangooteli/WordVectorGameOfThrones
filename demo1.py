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
#import gensim.models.word2vec as w2v
#import gensim

from gensim.models import Word2Vec

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




thrones2vec = Word2Vec.load(os.path.join("trained","thrones2vec.w2v"))

#thrones2vec.init_sims();

#compress the word vectors into 2D space and plot them
tsne = sklearn.manifold.TSNE(n_components = 2, random_state = 0)
all_word_vectors_matrix = thrones2vec.wv.syn0

print("train t-SNE, this could take a minute or two")
#train t-SNE, this could take a minute or two
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

#plot the big picture
points = pd.DataFrame(
	[
		(word, coords[0], coords[1])
		for word, coords in [
			(word, all_word_vectors_matrix_2d[thrones2vec.wv.vocab[word].index])
			for word in thrones2vec.wv.vocab
		]
	],
	columns = ['word', 'x', 'y']
)

print(points.head(10))

#sns.set_context('poster')
#points.plot.scatter('x', 'y', s=10, figsize=(20,12))


#def plot_region(x_bounds, y_bounds):
#	slice = points[
#		(x_bounds[0] <= points.x) &
#		(points.x <= x_bounds[1]) &
#		(y_bounds[0] <= points.y) &
#		(points.y <= y_bounds[1])
#	]
	
#	ax = slice.plot.scatter('x', 'y', s=35, figsize=(10, 8))
#	for i, point in slice.iterrows():
#		ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize = 11)

#plot_region(x_bounds = (4.0, 4.2), y_bounds = (-0.5, -0.1))

#plot_region(x_bounds = (0, 1), y_bounds = (4, 4.5))

#explore semantic similarities between book characters

#Word closest to the given word
#print(thrones2vec.most_similar('Stark'))
#print(thrones2vec.most_similar('Aerys'))
#print(thrones2vec.most_similar('direwolf'))

#Linear relationships between word pairs
#def nearest_similarity_cosmul(start1, end1, end2):
#	similarities = thrones2vec.wv.most_similar_cosmul(
#		positive = [end2, start1],
#		negative = [end1]
#	)
#	start2 = similarities[0][0]
#	print('{start1} is related to {end1}, as {start2} is related to {end2}'.format(**locals()))
#	return start2

#nearest_similarity_cosmul('Stark', 'Winterfell', 'Riverrun')
#nearest_similarity_cosmul('Jaime', 'sword', 'wine')
#nearest_similarity_cosmul('Arya', 'Nymeria', 'dragons')



def nearest_similarity_cosmul(start1, end1, end2):
    similarities = thrones2vec.most_similar_cosmul(
        positive = [end2, start1],
        negative = [end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2


nearest_similarity_cosmul("Stark", "Winterfell", "Riverrun")
nearest_similarity_cosmul("Jaime", "sword", "wine")
nearest_similarity_cosmul("Arya", "Nymeria", "dragons")
