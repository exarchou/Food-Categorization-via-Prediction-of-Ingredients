# =============================================================================
# Script to replace the invalid word2vec names of classes & ingredients
# Author: Dimitrios-Marios Exarchou
# Last modified: 11/6/2021   
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
current_data_dir   = 'X:/thesis/Task3'
output_data_dir    = 'X:/thesis_past/data'                # I have to copy it to 'X:/thesis_outputs/Task3'
image_data_dir     = 'X:/thesis_past/data/images/'        # I have to copy it to 'X:/datasets/recipe1m/images'
im2recipe_data_dir = 'X:/thesis_outputs/InverseCooking'
word2vec_data      = 'X:/thesis_outputs/Task3/word2vec_data'
output_dataset_dir = 'X:/thesis_outputs/Task3/final_dataset'
annoy_indexer_dir  = 'X:/thesis_outputs/Task3/annoy_index'

EMBEDDING_VECTOR_LENGTH = 300
MAX_SEQUENCE_LENGTH = 20

ingrs_vocab = pickle.load(open(os.path.join(im2recipe_data_dir, 'ingr_vocab.pkl'), 'rb'))
embedding_matrix = pickle.load(open(os.path.join(word2vec_data, 'embedding_matrix.pkl'), 'rb'))
id2class_reloaded  = pickle.load(open(os.path.join(output_dataset_dir, 'id2class_reloaded.pkl'), 'rb'))

X_train = pickle.load(open(os.path.join(output_dataset_dir, 'X_train.pkl'), 'rb'))
y_train = pickle.load(open(os.path.join(output_dataset_dir, 'y_train.pkl'), 'rb'))
X_valid = pickle.load(open(os.path.join(output_dataset_dir, 'X_valid.pkl'), 'rb'))
y_valid = pickle.load(open(os.path.join(output_dataset_dir, 'y_valid.pkl'), 'rb'))
X_test  = pickle.load(open(os.path.join(output_dataset_dir, 'X_test.pkl'), 'rb'))
y_test  = pickle.load(open(os.path.join(output_dataset_dir, 'y_test.pkl'), 'rb'))




#%% STEP 3: Find invalid classes
word2vec_model = KeyedVectors.load_word2vec_format(os.path.join(word2vec_data, 'GoogleNews-vectors-negative300.bin'), binary=True)

counter, invalid_classes, invalid_classes_num = 0, [], []

for i in tqdm(range(len(id2class_reloaded))):
    
    word = id2class_reloaded[i]  
    
    try:
        vec = word2vec_model[word]
        
    except:
        print("Invalid word:", word)
        counter += 1
        invalid_classes.append(word)
        invalid_classes_num.append(i)
        
print("Total invalids:", counter)


# Find frequencies of invalids classes
freq = np.zeros((len(invalid_classes_num),), dtype = float)
freq_dict = {}

for i in range(len(invalid_classes)):
    freq_dict[invalid_classes[i]] = 0


for i in tqdm(range(len(y_train))):  
    if y_train.iloc[i] in invalid_classes_num:
        freq[invalid_classes_num.index(y_train.iloc[i])] += 1
        freq_dict[id2class_reloaded[y_train.iloc[i]]] += 1
 
for i in tqdm(range(len(y_valid))): 
    if y_valid.iloc[i] in invalid_classes_num:  
        freq[invalid_classes_num.index(y_valid.iloc[i])] += 1
        freq_dict[id2class_reloaded[y_valid.iloc[i]]] += 1

for i in tqdm(range(len(y_test))): 
    if y_test.iloc[i] in invalid_classes_num:
        freq[invalid_classes_num.index(y_test.iloc[i])] += 1
        freq_dict[id2class_reloaded[y_test.iloc[i]]] += 1


for i in range(len(freq)):   
    freq[i] /= (len(y_train) + len(y_valid) + len(y_test))

# Sorting descending
freq = np.sort(freq)[::-1]


for key in freq_dict:
    freq_dict[key] /= (len(y_train) + len(y_valid) + len(y_test))


# Find the classes to find synonyms, with a rate of appearance over 0.001
classes_to_find_synonyms = []

for key in freq_dict:
    if freq_dict[key] > 0.001:
        classes_to_find_synonyms.append(key)
        
    


#%% STEP 4: Define an Annoy Indexer
model = word2vec_model
annoy_index = AnnoyIndexer()
annoy_index.load(annoy_indexer_dir)




#%% STEP 4: Replace manually top 61 (and more) invalid classes with some valid names
def replace_value(mydict, initial_value, final_value):    
    for key in mydict:     
        if mydict[key] == initial_value:
            mydict[key] = final_value
            break
            
    return mydict

# Initialization
id2class_custom = {}

for key in id2class_reloaded:
    id2class_custom[key] = id2class_reloaded[key]


# Replace the 61 most frequent invalid classes
id2class_custom = replace_value(id2class_custom, 'sugar_cookies', 'cookies')
id2class_custom = replace_value(id2class_custom, 'black_bean', 'bean')
id2class_custom = replace_value(id2class_custom, 'bread_machine', 'bread')
id2class_custom = replace_value(id2class_custom, 'white_chocolate', 'chocolate')
id2class_custom = replace_value(id2class_custom, 'whole_wheat', 'wheat')
id2class_custom = replace_value(id2class_custom, 'coffee_cake', 'chocolate_cake')
id2class_custom = replace_value(id2class_custom, 'blue_cheese', 'roquefort')
id2class_custom = replace_value(id2class_custom, 'red_pepper', 'pepper')
id2class_custom = replace_value(id2class_custom, 'pound_cake', 'cake')
id2class_custom = replace_value(id2class_custom, 'sugar_cookie', 'cookie')
id2class_custom = replace_value(id2class_custom, 'macaroni_and_cheese', 'macaroni') 
id2class_custom = replace_value(id2class_custom, 'chicken_pasta', 'chicken') 
id2class_custom = replace_value(id2class_custom, 'butter_cookies', 'cookies')
id2class_custom = replace_value(id2class_custom, 'white_bread', 'bread')
id2class_custom = replace_value(id2class_custom, 'key_lime', 'lime')
id2class_custom = replace_value(id2class_custom, 'hot_chocolate', 'chocolate')
id2class_custom = replace_value(id2class_custom, 'apple_cake', 'apple_pie')
id2class_custom = replace_value(id2class_custom, 'nut_bread', 'bread')
id2class_custom = replace_value(id2class_custom, 'green_been', 'green_beans')
id2class_custom = replace_value(id2class_custom, 'pumpkin_bread', 'bread')
id2class_custom = replace_value(id2class_custom, 'cooker_chicken', 'chicken')
id2class_custom = replace_value(id2class_custom, 'hot_dog', 'hotdog')
id2class_custom = replace_value(id2class_custom, 'cheese_soup', 'NOISE_CLASS')
id2class_custom = replace_value(id2class_custom, 'garlic_chicken', 'chicken')
id2class_custom = replace_value(id2class_custom, 'ground_beef', 'beef')
id2class_custom = replace_value(id2class_custom, 'white_bean', 'bean')
id2class_custom = replace_value(id2class_custom, 'sun-dried_tomato', 'dried_tomato')
id2class_custom = replace_value(id2class_custom, 'lemon_chicken', 'chicken')
id2class_custom = replace_value(id2class_custom, 'pulled_pork', 'pork')
id2class_custom = replace_value(id2class_custom, 'chocolate_covered', 'chocolate')
id2class_custom = replace_value(id2class_custom, 'wild_rice', 'rice')
id2class_custom = replace_value(id2class_custom, 'garlic_shrimp', 'shrimp_scampi')
id2class_custom = replace_value(id2class_custom, 'cake_mix', 'cake')
id2class_custom = replace_value(id2class_custom, 'dinner_rolls', 'french_bread')
id2class_custom = replace_value(id2class_custom, 'banana_cake', 'cake')
id2class_custom = replace_value(id2class_custom, 'monkey_bread', 'bread')
id2class_custom = replace_value(id2class_custom, 'upside-down_cake', 'cake')
id2class_custom = replace_value(id2class_custom, 'taco_soup', 'tortilla_soup')
id2class_custom = replace_value(id2class_custom, 'orange_chicken', 'chicken')
id2class_custom = replace_value(id2class_custom, 'cheese_ball', 'cheese')
id2class_custom = replace_value(id2class_custom, 'creamy_chicken', 'chicken')
id2class_custom = replace_value(id2class_custom, 'nut_muffins', 'muffins')
id2class_custom = replace_value(id2class_custom, 'red_wine', 'wine')
id2class_custom = replace_value(id2class_custom, 'cheesy_chicken', 'chicken')
id2class_custom = replace_value(id2class_custom, 'lime_chicken', 'chicken')
id2class_custom = replace_value(id2class_custom, 'double_chocolate', 'chocolate')
id2class_custom = replace_value(id2class_custom, 'pumpkin_cookies', 'cookies')
id2class_custom = replace_value(id2class_custom, 'cookie_bars', 'cookies')
id2class_custom = replace_value(id2class_custom, 'lemon_bars', 'bars')
id2class_custom = replace_value(id2class_custom, 'spinach_dip', 'spinach')
id2class_custom = replace_value(id2class_custom, 'bow_tie', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'stuffed_chicken', 'chicken')
id2class_custom = replace_value(id2class_custom, 'potato_bake', 'scalloped_potatoes')
id2class_custom = replace_value(id2class_custom, 'ranch_chicken', 'chicken')
id2class_custom = replace_value(id2class_custom, 'breakfast_casserole', 'casserole')
id2class_custom = replace_value(id2class_custom, 'stuffed_shells', 'shells')
id2class_custom = replace_value(id2class_custom, 'butter_chicken', 'chicken')
id2class_custom = replace_value(id2class_custom, 'weight_watchers', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'baked_salmon', 'salmon')
id2class_custom = replace_value(id2class_custom, 'cheese_bread', 'bread')
id2class_custom = replace_value(id2class_custom, 'pizza_sauce', 'pizza')


# Continue replacing
id2class_custom = replace_value(id2class_custom, 'chocolate_cookies', 'cookies')
id2class_custom = replace_value(id2class_custom, 'corn_salad', 'corn') 
id2class_custom = replace_value(id2class_custom, 'nut_cookies', 'cookies')
id2class_custom = replace_value(id2class_custom, 'red_onion', 'vegetable_medley')
id2class_custom = replace_value(id2class_custom, 'bacon_cheese', 'cheese')
id2class_custom = replace_value(id2class_custom, 'beer_bread', 'bread')
id2class_custom = replace_value(id2class_custom, 'green_pepper', 'pepper')
id2class_custom = replace_value(id2class_custom, 'chicken_chili', 'chicken')
id2class_custom = replace_value(id2class_custom, 'turkey_chili', 'turkey')
id2class_custom = replace_value(id2class_custom, 'corn_casserole', 'corn_chowder')
id2class_custom = replace_value(id2class_custom, 'apple_salad', 'apple')
id2class_custom = replace_value(id2class_custom, 'butter_cake', 'cake')
id2class_custom = replace_value(id2class_custom, 'mexican_rice', 'rice')
id2class_custom = replace_value(id2class_custom, 'balsamic_chicken', 'chicken')
id2class_custom = replace_value(id2class_custom, 'pasta_bake', 'pasta')
id2class_custom = replace_value(id2class_custom, 'italian_chicken', 'chicken')
id2class_custom = replace_value(id2class_custom, 'salsa_chicken', 'chicken')
id2class_custom = replace_value(id2class_custom, 'backed_eggs', 'eggs')
id2class_custom = replace_value(id2class_custom, 'back_ribs', 'ribs')
id2class_custom = replace_value(id2class_custom, 'schrimp_salad', 'schrimps')
id2class_custom = replace_value(id2class_custom, 'cheese_dip', 'cheese') 
id2class_custom = replace_value(id2class_custom, 'food_cake', 'cake')
id2class_custom = replace_value(id2class_custom, 'olive_garden', 'olives') #
id2class_custom = replace_value(id2class_custom, 'pumpkin_cake', 'cake')
id2class_custom = replace_value(id2class_custom, 'red_cabbage', 'cabbage')
id2class_custom = replace_value(id2class_custom, 'summer_squash', 'vegetables')
id2class_custom = replace_value(id2class_custom, 'rice_bowl', 'rice')
id2class_custom = replace_value(id2class_custom, 'rice_salad', 'rice')
id2class_custom = replace_value(id2class_custom, 'chicken_bake', 'chicken')
id2class_custom = replace_value(id2class_custom, 'grilled_corn', 'corn')
id2class_custom = replace_value(id2class_custom, 'steak_marinade', 'steak')
id2class_custom = replace_value(id2class_custom, 'glazed_chicken', 'chicken')
id2class_custom = replace_value(id2class_custom, 'green_onion', 'vegetable_medley')
id2class_custom = replace_value(id2class_custom, 'olive_garden', 'olives')
id2class_custom = replace_value(id2class_custom, 'red_beans', 'beans')
id2class_custom = replace_value(id2class_custom, 'stove_top', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'taco_dip', 'cheese')
id2class_custom = replace_value(id2class_custom, 'cream_cake', 'cake')
id2class_custom = replace_value(id2class_custom, 'chicken_spaghetti', 'spaghetti')
id2class_custom = replace_value(id2class_custom, 'roasted_red', 'peppers')
id2class_custom = replace_value(id2class_custom, 'red_potatoes', 'scalloped_potatoes')
id2class_custom = replace_value(id2class_custom, 'cheese_chicken', 'chicken')
id2class_custom = replace_value(id2class_custom, 'chicken_pizza', 'pizza')
id2class_custom = replace_value(id2class_custom, 'lemon_cake', 'cake')
id2class_custom = replace_value(id2class_custom, 'white_wine', 'wine')
id2class_custom = replace_value(id2class_custom, 'chicken_fingers', 'chicken')
id2class_custom = replace_value(id2class_custom, 'green_tomato', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'angel_hair', 'pasta')
id2class_custom = replace_value(id2class_custom, 'snack_mix', 'snacks')
id2class_custom = replace_value(id2class_custom, 'wine_sauce', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'short_ribs', 'meat')
id2class_custom = replace_value(id2class_custom, 'todd_wilbur', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'pesto_chicken', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'corn_bread', 'bread') 
id2class_custom = replace_value(id2class_custom, 'shrimp_pasta', 'shrimp_scampi')
id2class_custom = replace_value(id2class_custom, 'paula_deen', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'chicken_stir-fry', 'chicken')
id2class_custom = replace_value(id2class_custom, 'roll_ups', 'wine')
id2class_custom = replace_value(id2class_custom, 'white_wine', 'NOISE_CLASS')
id2class_custom = replace_value(id2class_custom, 'hot_cocoa', 'cocoa')
id2class_custom = replace_value(id2class_custom, 'green_tea', 'tea')
id2class_custom = replace_value(id2class_custom, 'christmas_cookies', 'cookies')
id2class_custom = replace_value(id2class_custom, 'meat_sauce', 'beaf')
id2class_custom = replace_value(id2class_custom, 'spice_cake', 'cake')
id2class_custom = replace_value(id2class_custom, 'baked_fish', 'fillets') #
id2class_custom = replace_value(id2class_custom, 'fruit_dip', 'fruits')
id2class_custom = replace_value(id2class_custom, 'fruit_salse', 'fruits')
id2class_custom = replace_value(id2class_custom, 'black-eyed_pea', 'beans')
id2class_custom = replace_value(id2class_custom, 'cheese_spread', 'cheese')
id2class_custom = replace_value(id2class_custom, 'banana_split', 'banana')
id2class_custom = replace_value(id2class_custom, 'side_dish', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'egg_sandwich', 'sandwich')
id2class_custom = replace_value(id2class_custom, 'yellow_squash', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'game_hens', 'chicken')
id2class_custom = replace_value(id2class_custom, 'rocky_road', 'chocolate')
id2class_custom = replace_value(id2class_custom, 'sour_chicken', 'chicken')
id2class_custom = replace_value(id2class_custom, 'meat_pie', 'pie')
id2class_custom = replace_value(id2class_custom, 'brown_butter', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'breakfast_sandwich', 'egg')
id2class_custom = replace_value(id2class_custom, 'chocolate_pie', 'chocolate_cake')
id2class_custom = replace_value(id2class_custom, 'apple_bread', 'apple_pie')
id2class_custom = replace_value(id2class_custom, 'cheese_stuffed', 'cheese')
id2class_custom = replace_value(id2class_custom, 'rosemary_chicken', 'chicken') # Stop at a frequency of 0.00049234


# Replace Noisy classes that I found AFTER Classification
id2class_custom = replace_value(id2class_custom, 'chicken_breast', 'chicken')
id2class_custom = replace_value(id2class_custom, 'chicken_skewers', 'chicken')
id2class_custom = replace_value(id2class_custom, 'chicken_parmesan', 'chicken')
id2class_custom = replace_value(id2class_custom, 'cream_cheese', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'crock_pot', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'sour_cream', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'chicken_breasts', 'chicken')
id2class_custom = replace_value(id2class_custom, 'wheat', 'bread')
id2class_custom = replace_value(id2class_custom, 'peppers', 'pepper')
id2class_custom = replace_value(id2class_custom, 'baked_chicken', 'chicken')
id2class_custom = replace_value(id2class_custom, 'fried_chicken', 'chicken')
id2class_custom = replace_value(id2class_custom, 'fried_rice', 'rice')
id2class_custom = replace_value(id2class_custom, 'rice_pilaf', 'rice')
id2class_custom = replace_value(id2class_custom, 'cookie', 'cookies')
id2class_custom = replace_value(id2class_custom, 'buffalo_chicken', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'brown_sugar', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'wine', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'cream_sauce', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'pie_crust', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'pizza_crust', 'pizza')
id2class_custom = replace_value(id2class_custom, 'roast_chicken', 'chicken')
id2class_custom = replace_value(id2class_custom, 'roasted_chicken', 'chicken')
id2class_custom = replace_value(id2class_custom, 'potato_casserole', 'scalloped_potatoes')
id2class_custom = replace_value(id2class_custom, 'beef', 'minced_meat')
id2class_custom = replace_value(id2class_custom, 'bbq_chicken', 'chicken')
id2class_custom = replace_value(id2class_custom, 'bundt_cake', 'cake')
id2class_custom = replace_value(id2class_custom, 'honey_mustard', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'chicken_thighs', 'chicken')
id2class_custom = replace_value(id2class_custom, 'spicy_chicken', 'chicken')


# Replace plurals and same classes
id2class_custom = replace_value(id2class_custom, 'barbecued_chicken', 'barbecue_chicken')
id2class_custom = replace_value(id2class_custom, 'bean', 'beans')
id2class_custom = replace_value(id2class_custom, 'bell_pepper', 'bell_peppers')
id2class_custom = replace_value(id2class_custom, 'breakfast_burrito', 'breakfast_burritos')
id2class_custom = replace_value(id2class_custom, 'curried_chicken', 'curry_chicken')

# Replace Noisy classes again
id2class_custom = replace_value(id2class_custom, 'pepper', 'vegetable_medley')
id2class_custom = replace_value(id2class_custom, 'pork_tenderloin', 'pork_chops')
id2class_custom = replace_value(id2class_custom, 'pork_tenderloin', 'pork_chops')
id2class_custom = replace_value(id2class_custom, 'grilled_chicken', 'chicken')
id2class_custom = replace_value(id2class_custom, 'buttercream_frosting', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'pizza_dough', 'pizza')
id2class_custom = replace_value(id2class_custom, 'parmesan_chicken', 'chicken')
id2class_custom = replace_value(id2class_custom, 'spaghetti_sauce', 'tomato_sauce')
id2class_custom = replace_value(id2class_custom, 'flank_steak', 'steak')
id2class_custom = replace_value(id2class_custom, 'buttermilk_pancakes', 'pancakes')
id2class_custom = replace_value(id2class_custom, 'whipped_cream', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'bran_muffins', 'muffins')
id2class_custom = replace_value(id2class_custom, 'blueberry_muffins', 'muffins')
id2class_custom = replace_value(id2class_custom, 'barbecue_sauce', 'bbq_sauce')
id2class_custom = replace_value(id2class_custom, 'roasted_vegetable', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'pot_roast', 'pork_roast')
id2class_custom = replace_value(id2class_custom, 'salmon', 'grilled_salmon')
id2class_custom = replace_value(id2class_custom, 'smoked_salmon', 'grilled_salmon')
id2class_custom = replace_value(id2class_custom, 'spanish_rice', 'rice')
id2class_custom = replace_value(id2class_custom, 'sweet_corn', 'corn')
id2class_custom = replace_value(id2class_custom, 'banana_smoothie', 'fruit_smoothie')
id2class_custom = replace_value(id2class_custom, 'butter_sauce', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'olive_oil', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'grilled_asparagus', 'roasted_asparagus')
id2class_custom = replace_value(id2class_custom, 'chicken_salad', 'caesar_salad')
id2class_custom = replace_value(id2class_custom, 'lemon_pepper', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'olive_oil', 'NOISE_CLASS') ## Noise
id2class_custom = replace_value(id2class_custom, 'spinach_pie', 'NOISE_CLASS') ### Too many problems for this class


# Save edited id2class dic
with open(os.path.join(output_dataset_dir, 'id2class_custom.pkl'), 'wb') as f:
    pickle.dump(id2class_custom, f)


# =============================================================================
# ## Code to plot an image from an ivalid class, to detect noisy classes
# inv_class_name = 'rosemary_chicken'
# 
# for key in id2class_reloaded:
#     if id2class_reloaded[key] ==  inv_class_name:
#         class_num = key
#         
# counter = 0
# indeces = []
# for i in range(len(Y_test_int)):
#     if Y_test_int[i] == class_num:
#         counter += 1
#         indeces.append(i)
#         if counter == 10:
#             break
#     
# for index in indeces:
#     image = Image.open(X_test_paths[index]).convert('RGB')
#     plt.figure()
#     plt.imshow(image)
#     plt.axis('off')
#     plt.show()
# 
# 
# print("Ingredients:")
# for index in indeces:
#     print("="*40)
#     for ing in X_test[index]:
#         if ing != 0:
#             print(ingr_vocab[ing])
#             
#     print("\nClass:", id2class_reloaded[Y_test_int[index]])
#     print("="*40)
# 
# # print(word2vec_model.most_similar([word2vec_model['apple_pie']], topn=81))
# =============================================================================


#%% Find the unique classes of id2class_custom
id2class_custom_valids = {}

for key in id2class_custom:
    id2class_custom_valids[key] = id2class_custom[key]

counter = 0
for key in id2class_custom:

    word = id2class_custom[key]
    try:
        vec = word2vec_model[word]
           
    except:
        id2class_custom_valids.pop(key)
        counter += 1
        
uniqueValues = set(id2class_custom_valids.values())
print("Total invalids:", counter)
print('Unique Values:', len(uniqueValues))

with open(os.path.join(output_dataset_dir, 'id2class_custom_valids.pkl'), 'wb') as f:
    pickle.dump(id2class_custom_valids, f)


# Find invalids after the replacement
counter = 0
invalid_classes = []
invalid_classes_num = []
word2vec_representations = []

for i in tqdm(range(len(id2class_custom))):

    word = id2class_custom[i]
    
    try:
        vec = word2vec_model[word]
        word2vec_representations.append(vec)
        
    except:
        print("Invalid word:", word)
        counter += 1
        invalid_classes.append(word)
        invalid_classes_num.append(i)
        word2vec_representations.append(np.zeros((EMBEDDING_VECTOR_LENGTH,), dtype=float))
        
print("Total invalids:", counter)

word2vec_representations = np.array(word2vec_representations)





#%% STEP 5: Create word2vec classes datasets Y

# Train Set
X_train_word2vec, y_train_word2vec = [], []

for i in tqdm(range(len(y_train))):
    
    if y_train.iloc[i] in invalid_classes_num: 
        continue   
    
    else:
        original_word = id2class_custom[y_train.iloc[i]]
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
        original_word = id2class_custom[y_valid.iloc[i]]
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
        original_word = id2class_custom[y_test.iloc[i]]
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



# Check freqs of classes
ans1 = y_train.value_counts() / len(y_train)
ans2 = y_valid.value_counts() / len(y_valid)
ans3 = y_test.value_counts()  / len(y_test)

