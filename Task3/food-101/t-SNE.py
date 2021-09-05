# =============================================================================
# Script to create t-SNE visualization for Food-101 Classes
# =============================================================================


#%% STEP 1: Libraries
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from gensim.similarities.annoy import AnnoyIndexer
from sklearn.manifold import TSNE




#%% STEP 2: Loading necessary files
# Directories
current_data_dir     = 'X:/thesis/Task3/food-101'
im2recipe_data_dir   = 'X:/thesis_outputs/InverseCooking'
word2vec_data        = 'X:/thesis_outputs/Task3/word2vec_data'
annoy_indexer_dir    = 'X:/thesis_outputs/Task3/annoy_index'
word2vec_dataset_dir = 'X:/thesis_outputs/Task3/food-101/word2vec_dataset'
save_output_dir      = 'X:/thesis_outputs/Task3/food-101/train_model'

EMBEDDING_VECTOR_LENGTH = 300
MAX_SEQUENCE_LENGTH = 20

ingr_vocab = pickle.load(open(os.path.join(im2recipe_data_dir, 'ingr_vocab.pkl'), 'rb'))
embedding_matrix = pickle.load(open(os.path.join(word2vec_data, 'embedding_matrix.pkl'), 'rb'))

X_train = pickle.load(open(os.path.join(word2vec_dataset_dir, 'X_train_word2vec.pkl'), 'rb'))
y_train = pickle.load(open(os.path.join(word2vec_dataset_dir, 'y_train_word2vec.pkl'), 'rb'))
X_valid = pickle.load(open(os.path.join(word2vec_dataset_dir, 'X_valid_word2vec.pkl'), 'rb'))
y_valid = pickle.load(open(os.path.join(word2vec_dataset_dir, 'y_valid_word2vec.pkl'), 'rb'))
X_test  = pickle.load(open(os.path.join(word2vec_dataset_dir, 'X_test_word2vec.pkl'), 'rb'))
y_test  = pickle.load(open(os.path.join(word2vec_dataset_dir, 'y_test_word2vec.pkl'), 'rb'))
y_test_int = np.loadtxt(os.path.join(word2vec_dataset_dir, 'y_test_int.txt'), dtype=int)

classes_reloaded  = pickle.load(open(os.path.join(word2vec_dataset_dir, 'classes_reloaded.pkl'), 'rb'))

word2vec_model = KeyedVectors.load_word2vec_format(os.path.join(word2vec_data, 'GoogleNews-vectors-negative300.bin'), binary=True)
annoy_index = AnnoyIndexer()
annoy_index.load(annoy_indexer_dir)






#%% t-SNE
labels = []
tokens = []


for word in classes_reloaded:
    tokens.append(word2vec_model[word])
    labels.append(word)

tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(tokens)

x = []
y = []
for value in new_values:
    x.append(value[0])
    y.append(value[1])
    
plt.figure(figsize=(16, 16)) 
for i in range(len(x)):
    plt.scatter(x[i],y[i])
    plt.annotate(labels[i],
                 xy=(x[i], y[i]),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
plt.axis('off')
plt.savefig(os.path.join(save_output_dir, 't-SNE.jpg'), transparent=True, dpi=200)
plt.show()



