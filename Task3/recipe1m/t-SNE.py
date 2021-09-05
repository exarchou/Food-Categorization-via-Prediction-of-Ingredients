
# =============================================================================
# Script to create t-SNE visualization for Recipe1M Classes
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
current_data_dir   = 'X:/thesis/Task3'
output_data_dir    = 'X:/thesis_past/data'                # I have to copy it to 'X:/thesis_outputs/Task3'
image_data_dir     = 'X:/thesis_past/data/images/'        # I have to copy it to 'X:/datasets/recipe1m/images'
im2recipe_data_dir = 'X:/thesis_outputs/InverseCooking'
word2vec_data      = 'X:/thesis_outputs/Task3/word2vec_data'
output_dataset_dir = 'X:/thesis_outputs/Task3/final_dataset'
annoy_indexer_dir  = 'X:/thesis_outputs/Task3/annoy_index'
save_output_dir    = 'X:/thesis_outputs/Task3/train_model_300_cells'

EMBEDDING_VECTOR_LENGTH = 300
MAX_SEQUENCE_LENGTH = 20

ingr_vocab = pickle.load(open(os.path.join(im2recipe_data_dir, 'ingr_vocab.pkl'), 'rb'))
embedding_matrix = pickle.load(open(os.path.join(word2vec_data, 'embedding_matrix.pkl'), 'rb'))
id2class_reloaded  = pickle.load(open(os.path.join(output_dataset_dir, 'id2class_reloaded.pkl'), 'rb'))
id2class_custom  = pickle.load(open(os.path.join(output_dataset_dir, 'id2class_custom.pkl'), 'rb'))
id2class_custom_valids  = pickle.load(open(os.path.join(output_dataset_dir, 'id2class_custom_valids.pkl'), 'rb'))

X_train = pickle.load(open(os.path.join(output_dataset_dir, 'X_train_word2vec.pkl'), 'rb'))
y_train = pickle.load(open(os.path.join(output_dataset_dir, 'y_train_word2vec.pkl'), 'rb'))
X_valid = pickle.load(open(os.path.join(output_dataset_dir, 'X_valid_word2vec.pkl'), 'rb'))
y_valid = pickle.load(open(os.path.join(output_dataset_dir, 'y_valid_word2vec.pkl'), 'rb'))
X_test  = pickle.load(open(os.path.join(output_dataset_dir, 'X_test_word2vec.pkl'), 'rb'))
y_test  = pickle.load(open(os.path.join(output_dataset_dir, 'y_test_word2vec.pkl'), 'rb'))
y_test_int = np.loadtxt(os.path.join(output_dataset_dir, 'y_test_int.txt'), dtype=int)

word2vec_model = KeyedVectors.load_word2vec_format(os.path.join(word2vec_data, 'GoogleNews-vectors-negative300.bin'), binary=True)
annoy_index = AnnoyIndexer()
annoy_index.load(annoy_indexer_dir)




#%% t-SNE
"Creates TSNE model and plots it"
labels = []
tokens = []

class_names = id2class_custom_valids.values()
class_names_unique = set(class_names)
class_names_unique = list(class_names_unique)

for word in class_names_unique:
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
plt.show()

