# =============================================================================
# Script to split dataset of ingrs-class equally regarding class into train/val/test sets for food-101
# Author: Dimitrios-Marios Exarchou
# Last modified: 22/8/2021   
# =============================================================================



#%% STEP 1: Import Libraries
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



#%% STEP 2: Loading necessary files
# Directories
current_data_dir   = 'X:/thesis/Task3.food-101'
im2recipe_data_dir = 'X:/thesis_outputs/InverseCooking'
dataset_dir        = 'X:/thesis_outputs/Task3/food-101/dataset'
word2vec_data      = 'X:/thesis_outputs/Task3/word2vec_data'
output_dataset_dir = 'X:/thesis_outputs/Task3/food-101/dataset'

ingrs_and_class = pickle.load(open(os.path.join(dataset_dir, 'ingrs_and_class.txt'), 'rb'))

ingrs_vocab = pickle.load(open(os.path.join(im2recipe_data_dir, 'ingr_vocab.pkl'), 'rb'))
embedding_matrix = pickle.load(open(os.path.join(word2vec_data, 'embedding_matrix.pkl'), 'rb'))


# Get rid of one-hot encoding
X = np.zeros((len(ingrs_and_class), 20), dtype=int)
y = np.zeros((len(ingrs_and_class), ), dtype=int)

for i in tqdm(range(len(ingrs_and_class))):
    row = ingrs_and_class[i]
    ingredients = row['ingredients']
     
    for j in range(len(ingredients)):
        X[i][j] = ingredients[j]
            
    y[i] = row['class']





#%% STEP 3: Concatanate the three splits, shuffle them and split again for equal porpotions

# Create DataFrame
dataset = np.append(X, np.expand_dims(y, axis=1), axis=1)
del(X, y)

df = pd.DataFrame(dataset, columns = ['ingr1','ingr2','ingr3','ingr4','ingr5','ingr6','ingr7','ingr8','ingr9','ingr10','ingr11','ingr12','ingr13','ingr14','ingr15','ingr16','ingr17','ingr18','ingr19','ingr20','label'])
#X, y = shuffle(X, y, random_state=42)

# Shuffle 
df = df.sample(frac=1, random_state=42)

X = df.drop('label', axis=1)
y = df.label

# Split into partitions equally: train: 70%, valid: 15%, test: 15%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)




#%% Save the dataframes and the id2class_reloaded 
with open(os.path.join(output_dataset_dir, 'X_train.pkl'), 'wb') as f:
    pickle.dump(X_train, f)

with open(os.path.join(output_dataset_dir, 'X_valid.pkl'), 'wb') as f:
    pickle.dump(X_valid, f)
    
with open(os.path.join(output_dataset_dir, 'X_test.pkl'), 'wb') as f:
    pickle.dump(X_test, f)

with open(os.path.join(output_dataset_dir, 'y_train.pkl'), 'wb') as f:
    pickle.dump(y_train, f)

with open(os.path.join(output_dataset_dir, 'y_valid.pkl'), 'wb') as f:
    pickle.dump(y_valid, f)
    
with open(os.path.join(output_dataset_dir, 'y_test.pkl'), 'wb') as f:
    pickle.dump(y_test, f)





#%% Load the dataframes
with open(os.path.join(output_dataset_dir, 'X_train.pkl'), 'rb') as f:
    X_train = pickle.load(f)

with open(os.path.join(output_dataset_dir, 'X_valid.pkl'), 'rb') as f:
    X_valid = pickle.load(f)
    
with open(os.path.join(output_dataset_dir, 'X_test.pkl'), 'rb') as f:
    X_test = pickle.load(f)
    
with open(os.path.join(output_dataset_dir, 'y_train.pkl'), 'rb') as f:
    y_train = pickle.load(f)

with open(os.path.join(output_dataset_dir, 'y_valid.pkl'), 'rb') as f:
    y_valid = pickle.load(f)
    
with open(os.path.join(output_dataset_dir, 'y_test.pkl'), 'rb') as f:
    y_test = pickle.load(f)




# freqs of classes to test for equal splitting
ans  = y.value_counts() / len(y)
ans1 = y_train.value_counts() / len(y_train)
ans2 = y_valid.value_counts() / len(y_valid)
ans3 = y_test.value_counts()  / len(y_test)




