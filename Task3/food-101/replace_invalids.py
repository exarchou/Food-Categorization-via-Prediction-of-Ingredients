# =============================================================================
# Script to replace the invalid word2vec names of classes & ingredients for Food-101 dataset
# Author: Dimitrios-Marios Exarchou
# Last modified: 22/8/2021   
# =============================================================================


#%% STEP 1: Import Libraries
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from gensim.models import KeyedVectors
from gensim.similarities.annoy import AnnoyIndexer



#%% STEP 2: Loading necessary files
# Directories
current_data_dir   = 'X:/thesis/Task3/food-101'

im2recipe_data_dir = 'X:/thesis_outputs/InverseCooking'
word2vec_data      = 'X:/thesis_outputs/Task3/word2vec_data'
dataset_dir        = 'X:/thesis_outputs/Task3/food-101/dataset'
output_dataset_dir = 'X:/thesis_outputs/Task3/food-101/word2vec_dataset'
annoy_indexer_dir  = 'X:/thesis_outputs/Task3/annoy_index'

EMBEDDING_VECTOR_LENGTH = 300
MAX_SEQUENCE_LENGTH = 20

ingrs_vocab = pickle.load(open(os.path.join(im2recipe_data_dir, 'ingr_vocab.pkl'), 'rb'))
embedding_matrix = pickle.load(open(os.path.join(word2vec_data, 'embedding_matrix.pkl'), 'rb'))

with open('X:/datasets/food-101//meta/classes.txt', "r") as f:
    classes = []
    for line in f: 
        classes.append(line[:len(line)-1])

X_train = pickle.load(open(os.path.join(dataset_dir, 'X_train.pkl'), 'rb'))
y_train = pickle.load(open(os.path.join(dataset_dir, 'y_train.pkl'), 'rb'))
X_valid = pickle.load(open(os.path.join(dataset_dir, 'X_valid.pkl'), 'rb'))
y_valid = pickle.load(open(os.path.join(dataset_dir, 'y_valid.pkl'), 'rb'))
X_test  = pickle.load(open(os.path.join(dataset_dir, 'X_test.pkl'), 'rb'))
y_test  = pickle.load(open(os.path.join(dataset_dir, 'y_test.pkl'), 'rb'))




#%% STEP 3: Find invalid classes
word2vec_model = KeyedVectors.load_word2vec_format(os.path.join(word2vec_data, 'GoogleNews-vectors-negative300.bin'), binary=True)
model = word2vec_model
annoy_index = AnnoyIndexer()
annoy_index.load(annoy_indexer_dir)

counter, invalid_classes, invalid_classes_num = 0, [], []

for i in tqdm(range(len(classes))):
    
    word = classes[i]  
    
    try:
        vec = word2vec_model[word]
        
    except:
        print("Invalid word:", word)
        counter += 1
        invalid_classes.append(word)
        invalid_classes_num.append(i)
        
print("\nTotal invalids:", counter)

    


#%% STEP 4: Manual Replacement
print('\nManual Replacement...\n')
classes[classes.index('baby_back_ribs')] = 'ribs'
classes[classes.index('cheese_plate')] = 'cheese'
classes[classes.index('club_sandwich')] = 'sandwich'
classes[classes.index('cup_cakes')] = 'cupcake'
classes[classes.index('fish_and_chips')] = 'fish'
classes[classes.index('hot_and_sour_soup')] = 'soup'
classes[classes.index('hot_dog')] = 'hotdog'
classes[classes.index('lobster_roll_sandwich')] = 'lobster'
classes[classes.index('macaroni_and_cheese')] = 'macaroni'
classes[classes.index('pulled_pork_sandwich')] = 'pork_sandwich'
classes[classes.index('red_velvet_cake')] = 'velvet_cake'
classes[classes.index('shrimp_and_grits')] = 'shrimps'
classes[classes.index('spring_rolls')] = 'rolls'


counter, invalid_classes, invalid_classes_num = 0, [], []

for i in tqdm(range(len(classes))):
    
    word = classes[i]  
    
    try:
        vec = word2vec_model[word]
        
    except:
        print("Invalid word:", word)
        counter += 1
        invalid_classes.append(word)
        invalid_classes_num.append(i)
        
print("\nTotal invalids:", counter)



#%% STEP 5: Create word2vec classes dataset

# Train Set
X_train_word2vec, y_train_word2vec = [], []

for i in tqdm(range(len(y_train))):
    
    if y_train.iloc[i] in invalid_classes_num: 
        continue   
    
    else:
        original_word = classes[y_train.iloc[i]]
        vec = word2vec_model[original_word]
        y_train_word2vec.append(vec)
        X_train_word2vec.append(X_train.iloc[i])
    
X_train_word2vec = pd.DataFrame(np.array(X_train_word2vec))
y_train_word2vec = pd.DataFrame(np.array(y_train_word2vec))


# Validation Set
X_valid_word2vec, y_valid_word2vec = [], []

for i in tqdm(range(len(y_valid))):
    
    if y_valid.iloc[i] in invalid_classes_num: 
        continue   
    
    else:
        original_word = classes[y_valid.iloc[i]]
        vec = word2vec_model[original_word]
        y_valid_word2vec.append(vec)
        X_valid_word2vec.append(X_valid.iloc[i])
    
X_valid_word2vec = pd.DataFrame(np.array(X_valid_word2vec))
y_valid_word2vec = pd.DataFrame(np.array(y_valid_word2vec))


# Test Set
X_test_word2vec, y_test_word2vec, y_test_int = [], [], []

for i in tqdm(range(len(y_test))):
    
    if y_test.iloc[i] in invalid_classes_num: 
        continue   
    
    else:
        original_word = classes[y_test.iloc[i]]
        vec = word2vec_model[original_word]
        y_test_word2vec.append(vec)
        y_test_int.append(y_test.iloc[i])
        X_test_word2vec.append(X_test.iloc[i])
    
X_test_word2vec = pd.DataFrame(np.array(X_test_word2vec))
y_test_word2vec = pd.DataFrame(np.array(y_test_word2vec))
y_test_int = np.array(y_test_int)



# Save the word2vec datasets
with open(os.path.join(output_dataset_dir, 'X_train_word2vec.pkl'), 'wb') as f:
    pickle.dump(X_train_word2vec, f)
    
with open(os.path.join(output_dataset_dir, 'X_valid_word2vec.pkl'), 'wb') as f:
    pickle.dump(X_valid_word2vec, f)
    
with open(os.path.join(output_dataset_dir, 'X_test_word2vec.pkl'), 'wb') as f:
    pickle.dump(X_test_word2vec, f)

with open(os.path.join(output_dataset_dir, 'y_train_word2vec.pkl'), 'wb') as f:
    pickle.dump(y_train_word2vec, f)
    
with open(os.path.join(output_dataset_dir, 'y_valid_word2vec.pkl'), 'wb') as f:
    pickle.dump(y_valid_word2vec, f)
    
with open(os.path.join(output_dataset_dir, 'y_test_word2vec.pkl'), 'wb') as f:
    pickle.dump(y_test_word2vec, f)  

np.savetxt(os.path.join(output_dataset_dir, 'y_test_int.txt'),  y_test_int, fmt='%d')

# Save the reloaded classes list
with open(os.path.join(output_dataset_dir, 'classes_reloaded.pkl'), 'wb') as f:
    pickle.dump(classes, f)



# Check freqs of classes
ans1 = y_train.value_counts() / len(y_train)
ans2 = y_valid.value_counts() / len(y_valid)
ans3 = y_test.value_counts()  / len(y_test)

