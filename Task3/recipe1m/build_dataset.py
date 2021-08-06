# =============================================================================
# Script to generate the dataset of ingredients-class and the embedding matrix of ingredients
# Author: Dimitrios-Marios Exarchou
# Last modified: 21/3/2021   
# Creating dataset: DONE
# Creating embedding matrix: DONE (583/1488 Invalids)
# =============================================================================



#%% STEP 1: Libraries and Loading files
import os
import time
import json
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from gensim.models import KeyedVectors
from InverseCooking.src.args import get_parser
from InverseCooking.src.model import get_model


# Directories
current_data_dir   = 'X:/thesis/Task3'
output_data_dir    = 'X:/thesis_past/data'                # I have to copy it to 'X:/thesis_outputs/Task3'
image_data_dir     = 'X:/thesis_past/data/images/'        # I have to copy it to 'X:/datasets/recipe1m/images'
im2recipe_data_dir = 'X:/thesis_outputs/InverseCooking'
word2vec_data      = 'X:/thesis_outputs/Task3/word2vec_data'


# Check if a GPU is available
use_gpu = True
device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
map_loc = None if torch.cuda.is_available() and use_gpu else 'cpu'

# Load necessary files
ingrs_vocab = pickle.load(open(os.path.join(im2recipe_data_dir, 'ingr_vocab.pkl'), 'rb'))
vocab = pickle.load(open(os.path.join(im2recipe_data_dir, 'instr_vocab.pkl'), 'rb'))

ingr_vocab_size = len(ingrs_vocab)
instrs_vocab_size = len(vocab)
output_dim = instrs_vocab_size

# Load model
t = time.time()
import sys; sys.argv=['']; del sys
args = get_parser()
args.maxseqlen = 15
args.ingrs_only=False
model = get_model(args, ingr_vocab_size, instrs_vocab_size)
# Load the trained model parameters
model_path = os.path.join(im2recipe_data_dir, 'modelbest.ckpt')
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()
model.ingrs_only = False
model.recipe_only = False
print ('loaded model')
print ("Elapsed time:", time.time() -t)

# Deconstruct model into its modules
modules = list(model.children())
ingr_encoder  = modules[0]
instr_decoder = modules[1]
img_encoder   = modules[2].to(device)
ingr_decoder  = modules[3].to(device)




#%% STEP 2: Loading data files
print ("Loading layers...")
layer1 = json.load(open(os.path.join(im2recipe_data_dir, 'layer1.json'), 'r'))
layer2 = json.load(open(os.path.join(im2recipe_data_dir, 'layer2.json'), 'r'))
ingrs_vocab = pickle.load(open(os.path.join(im2recipe_data_dir, 'ingr_vocab.pkl'), 'rb'))

with open(os.path.join(im2recipe_data_dir, 'classes1M.pkl'),'rb') as f:
    class_dict = pickle.load(f)
    id2class = pickle.load(f)
    
    


#%% STEP 3: Create the dataset of image_paths - class
X_train, Y_train = [], []
X_val, Y_val = [], []
X_test, Y_test = [], []

for entry_l1 in tqdm(layer1):
    
    if (class_dict[entry_l1['id']] != 0):
        
        for entry_l2 in layer2:
            
            if (entry_l2['id'] == entry_l1['id']):
                
                if (entry_l1['partition'] == 'train'):
                    
                    for img in entry_l2['images']:
                        
                        X_train.append(image_data_dir + 'train/' + img['id'][0] + '/' + img['id'][1] + '/' + img['id'][2] + '/' + img['id'][3] + '/' + img['id'])
                        Y_train.append(class_dict[entry_l1['id']])

                elif (entry_l1['partition'] == 'val'):
                    
                    for img in entry_l2['images']:
                        
                        X_val.append(image_data_dir + 'val/' + img['id'][0] + '/' + img['id'][1] + '/' + img['id'][2] + '/' + img['id'][3] + '/' + img['id'])
                        Y_val.append(class_dict[entry_l1['id']])

                else:
                    
                    for img in entry_l2['images']:
                        
                        X_test.append(image_data_dir + 'test/' + img['id'][0] + '/' + img['id'][1] + '/' + img['id'][2] + '/' + img['id'][3] + '/' + img['id'])
                        Y_test.append(class_dict[entry_l1['id']])



# ============================================================================= SAVE/LOAD
# # Save the dataset of image_paths - class
# with open(os.path.join(output_data_dir, 'image_paths-class/X_train.pkl'), 'wb') as f:
#     pickle.dump(X_train, f)
#     
# with open(os.path.join(output_data_dir, 'image_paths-class/Y_train.pkl'), 'wb') as f:
#     pickle.dump(Y_train, f)
# 
# with open(os.path.join(output_data_dir, 'image_paths-class/X_val.pkl'), 'wb') as f:
#     pickle.dump(X_val, f)
#     
# with open(os.path.join(output_data_dir, 'image_paths-class/Y_val.pkl'), 'wb') as f:
#     pickle.dump(Y_val, f)
# 
# with open(os.path.join(output_data_dir, 'image_paths-class/X_test.pkl'), 'wb') as f:
#     pickle.dump(X_test, f)
#     
# with open(os.path.join(output_data_dir, 'image_paths-class/Y_test.pkl'), 'wb') as f:
#     pickle.dump(Y_test, f)
# 
# 
# # Load the dataset of image_paths - class
# with open(os.path.join(output_data_dir, 'image_paths-class/X_train.pkl'), 'rb') as f:
#     X_train = pickle.load(f)
#     
# with open(os.path.join(output_data_dir, 'image_paths-class/Y_train.pkl'), 'rb') as f:
#     Y_train = pickle.load(f)
# 
# with open(os.path.join(output_data_dir, 'image_paths-class/X_val.pkl'), 'rb') as f:
#     X_val = pickle.load(f)
#     
# with open(os.path.join(output_data_dir, 'image_paths-class/Y_val.pkl'), 'rb') as f:
#     Y_val = pickle.load(f)
# 
# with open(os.path.join(output_data_dir, 'image_paths-class/X_test.pkl'), 'rb') as f:
#     X_test = pickle.load(f)
#     
# with open(os.path.join(output_data_dir, 'image_paths-class/Y_test.pkl'), 'rb') as f:
#     Y_test = pickle.load(f)
# =============================================================================




#%% STEP 4: Create the dataset ingredients - class via model predictions
transform = transforms.Compose([
                                transforms.Resize(size=256),
                                transforms.CenterCrop(size=224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])
                                ])


def label2onehot(my_class):
    my_array = np.zeros((len(id2class),), dtype=int)  
    my_array[my_class] = 1  
    return my_array


# Train set
X_train_ingrs = np.zeros((len(X_train), 20), dtype = int)
Y_train_ingrs = np.zeros((len(Y_train), len(id2class)), dtype = int)

for i in tqdm(range(len(X_train))):
    
    image_path = X_train[i]
    image = Image.open(image_path).convert('RGB')   
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    features = img_encoder(image_tensor)    
    ingr_ids, ingr_logits = ingr_decoder.sample(None, None, greedy=True, temperature=1.0, img_features=features, first_token_value=0, replacement=False) 

    X_train_ingrs[i] = np.array(ingr_ids.cpu())
    Y_train_ingrs[i] = label2onehot(Y_train[i])


# Validation set
X_val_ingrs = np.zeros((len(X_val), 20), dtype = int)
Y_val_ingrs = np.zeros((len(Y_val), len(id2class)), dtype = int)

for i in tqdm(range(len(X_val))):
    
    image_path = X_val[i]
    image = Image.open(image_path).convert('RGB')   
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    features = img_encoder(image_tensor)    
    ingr_ids, ingr_logits = ingr_decoder.sample(None, None, greedy=True, temperature=1.0, img_features=features, first_token_value=0, replacement=False) 

    X_val_ingrs[i] = np.array(ingr_ids.cpu())
    Y_val_ingrs[i] = label2onehot(Y_val[i])

    
# Test set
X_test_ingrs = np.zeros((len(X_test), 20), dtype = int)
Y_test_ingrs = np.zeros((len(Y_test), len(id2class)), dtype = int)

for i in tqdm(range(len(X_test))):
    
    image_path = X_test[i]
    image = Image.open(image_path).convert('RGB')   
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    features = img_encoder(image_tensor)    
    ingr_ids, ingr_logits = ingr_decoder.sample(None, None, greedy=True, temperature=1.0, img_features=features, first_token_value=0, replacement=False) 

    X_test_ingrs[i] = np.array(ingr_ids.cpu())
    Y_test_ingrs[i] = label2onehot(Y_test[i])



# =============================================================================
# # Save the dataset of ingrs - class
# # with open(os.path.join(output_data_dir, 'ingrs-class/X_train_ingrs.pkl'), 'wb') as f:
# #     pickle.dump(X_train_ingrs, f)
# 
# # with open(os.path.join(output_data_dir, 'ingrs-class/Y_train_ingrs.pkl'), 'wb') as f:
# #     pickle.dump(Y_train_ingrs, f)
#         
# # with open(os.path.join(output_data_dir, 'ingrs-class/X_val_ingrs.pkl'), 'wb') as f:
# #     pickle.dump(X_val_ingrs, f)
# 
# # with open(os.path.join(output_data_dir, 'ingrs-class/Y_val_ingrs.pkl'), 'wb') as f:
# #     pickle.dump(Y_val_ingrs, f)
# 
# # with open(os.path.join(output_data_dir, 'ingrs-class/X_test_ingrs.pkl'), 'wb') as f:
# #     pickle.dump(X_test_ingrs, f)
# 
# # with open(os.path.join(output_data_dir, 'ingrs-class/Y_test_ingrs.pkl'), 'wb') as f:
# #     pickle.dump(Y_test_ingrs, f)
# 
# 
# # Load the dataset of ingrs - class
# with open(os.path.join(output_data_dir, 'ingrs-class/X_train_ingrs.pkl'), 'rb') as f:
#     X_train = pickle.load(f)
#     
# with open(os.path.join(output_data_dir, 'ingrs-class/Y_train_ingrs.pkl'), 'rb') as f:
#     Y_train = pickle.load(f)
# 
# with open(os.path.join(output_data_dir, 'ingrs-class/X_val_ingrs.pkl'), 'rb') as f:
#     X_val = pickle.load(f)
#     
# with open(os.path.join(output_data_dir, 'ingrs-class/Y_val_ingrs.pkl'), 'rb') as f:
#     Y_val = pickle.load(f)
# 
# with open(os.path.join(output_data_dir, 'ingrs-class/X_test_ingrs.pkl'), 'rb') as f:
#     X_test = pickle.load(f)
#     
# with open(os.path.join(output_data_dir, 'ingrs-class/Y_test_ingrs.pkl'), 'rb') as f:
#     Y_test = pickle.load(f)
# =============================================================================





#%% STEP 5: Create the embedding matrix from word2vec model
word2vec_model = KeyedVectors.load_word2vec_format(os.path.join(word2vec_data, 'GoogleNews-vectors-negative300.bin'), binary=True)

# Word2vec parameters
EMBEDDING_VECTOR_LENGTH = 300 # <=300
embedding_dict = {}
counter_invalids = 0

for word in ingrs_vocab:
    
    try:
        vector = word2vec_model[word]
        
    except:
        vector = np.zeros((300,), dtype = float)
        counter_invalids += 1
        print("Invalid word:", word)
        
    embedding_dict[word] = vector
    
print("\nTotal invalids:", counter_invalids)


embedding_matrix=np.zeros((len(ingrs_vocab), EMBEDDING_VECTOR_LENGTH))

for i in range(len(ingrs_vocab)):   
    vector = embedding_dict[ingrs_vocab[i]] 
    embedding_matrix[i] = vector[:EMBEDDING_VECTOR_LENGTH]



with open(os.path.join(word2vec_data, 'embedding_matrix.pkl'), 'wb') as f:
    pickle.dump(embedding_matrix, f)


embedding_matrix = pickle.load(open(os.path.join(word2vec_data, 'embedding_matrix.pkl'), 'rb'))


