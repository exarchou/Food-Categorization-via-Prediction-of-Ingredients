#%% STEP 0: Import Libraries, Load necessary files and Create Dataloaders
# Import Libraries
import os
import pickle
import torch
import heapq
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt
from tqdm import tqdm



# Specify data directoris for loading
image_data_dir = 'X:/datasets/food-101/images'
current_data_dir = 'X:/thesis/Task2'
im2recipe_data_dir = 'X:/thesis_outputs/InverseCooking'
output_data_dir = 'X:/thesis_outputs'

# Specify type of dataset (simple_dataset / augmented_dataset)
type_of_dataset = 'simple_dataset'



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

f = open("X:/datasets/food-101//meta/classes.txt", "r")
classes = []
for line in f: 
    classes.append(line[:len(line)-1])








#%% STEP 1: LOAD ingrs_and_classes and ingrs_per_class datasets
ingrs_and_class = pickle.load(open(os.path.join(output_data_dir, 'Task2/' + type_of_dataset + '/ingrs_and_class.pkl'), 'rb'))
ingrs_per_class = pickle.load(open(os.path.join(output_data_dir, 'Task2/' + type_of_dataset + '/ingrs_per_class.pkl'), 'rb'))

# Print some stats    
sums_vector = ingrs_per_class.sum(axis=0)
print("Ingredients used:", np.count_nonzero(sums_vector))

counter = 0.0
for i in range(len(ingrs_and_class)):
    counter += np.count_nonzero(ingrs_and_class[i]['ingredients'])-1 # We substract 1 because of the <end> banner
print("Average Ingredients per image:",(float)(counter/len(ingrs_and_class)))


# Define a class for the ingrs_and_class pairs dataset
class Food101Ingredients(data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        """Returns one data pair (ingredients and class)."""
        sample = self.dataset[index]
        my_ingredients = sample['ingredients']
        my_class = sample['class']
        return my_ingredients.astype(np.float), my_class

    def __len__(self):
        return len(self.dataset)


IngredientsClassDataset = Food101Ingredients(ingrs_and_class)


# Define Dataloaders
batch_size = 128

train_set, val_set = torch.utils.data.random_split(IngredientsClassDataset, [(int)(len(IngredientsClassDataset)*0.8), (int)(len(IngredientsClassDataset)*0.2)])
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, num_workers=0)




#%% STEP 2: Predict Class with Possibilities Theory


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # Pause a bit so that plots are updated
    

def label2onehot(list): 
    my_array = np.zeros((1488,), dtype=int)   
    for i in list:
        my_array[i] = 1   
    return my_array



# ============================================================================= 
# my_images, my_classes = next(iter(testloader))
# print(my_images.shape, my_classes.shape)
# 
# 
# torch.cuda.empty_cache()
# my_images = my_images.to(device)
# my_classes = my_classes.to(device)
# 
# img_encoder = img_encoder.to(device)
# ingr_decoder = ingr_decoder.to(device)
# 
# 
# with torch.no_grad():
#     
#     features = img_encoder(my_images)
#     print(features.shape)
#     ingr_ids, ingr_logits = ingr_decoder.sample(None, None, greedy=True,
#                                                 temperature=1.0, img_features=features,
#                                                 first_token_value=0, replacement=False)
#     
# print(ingr_ids.shape, ingr_logits.shape)
# 
# for j in range(4):
#     
#     imshow(my_images[j], classes[my_classes[j]])
#     print("="*20)
#     for i in range(len(ingr_ids[j])):     
#         if ingr_ids[j][i] != 0: 
#             print(ingrs_vocab[ingr_ids[j][i]]) 
#     print("="*20)
# 
# print()
# print()
# print()
# =============================================================================





# Calculate the sums per column vector: namely the total appearances of each ingredient 
sums_vector = ingrs_per_class.sum(axis=0)

# Calculate a table of psossibilities: If we find a specific ingredient then it belongs to a certain class
ingrs_per_class_poss = np.zeros((ingrs_per_class.shape[0], ingrs_per_class.shape[1]), dtype=float)

for i in tqdm(range(ingrs_per_class.shape[0])):    
    for j in range(ingrs_per_class.shape[1]):    
        ingrs_per_class_poss[i][j] = (float) (ingrs_per_class[i][j] / sums_vector[j])




# # Save possibilities
# with open(os.path.join(output_data_dir, 'Task2/' + type_of_dataset + '/ingrs_per_class_poss.pkl'), 'wb') as fp:   
#     pickle.dump(ingrs_per_class_poss, fp)

# # Load possibilities
# ingrs_per_class_poss = pickle.load(open(os.path.join(output_data_dir, 'Task2/' + type_of_dataset + '/ingrs_per_class_poss.pkl'), 'rb'))




#%% Predicting step: for each input image we create an empty vector of 101 cells. Each cell
#   corresponds to the possibility of each class. For each of the predcted ingredients, a 
#   score is summed up in the corresponding class cell.

top1_accuracy, top5_accuracy = 0, 0

for entry in tqdm(ingrs_and_class):
    
    pred_ingredients, real_class = entry['ingredients'], entry['class']
    class_scores = np.zeros((101,), dtype=float)
    
    for index in range(len(pred_ingredients)):
        for possible_class in range(ingrs_per_class_poss.shape[0]):
            
            if pred_ingredients[index] == True:
                class_scores[possible_class] += ingrs_per_class_poss[possible_class][index]
        
    top5_classes = heapq.nlargest(5, range(len(class_scores)), class_scores.take)
        
    # Calculate top-1 Accuracy
    if top5_classes[0] == real_class:
        top1_accuracy += 1
        
    # Calculate top-5 Accuracy
    for pred_class in top5_classes:
        if pred_class == real_class:
            top5_accuracy += 1
            break




# =============================================================================
# accuracy = 0
# 
# for each_image in tqdm(range(my_images.shape[0])):
# 
#     image_from_batch = each_image
#     class_scores = np.zeros((101,), dtype=float)
#     
#     for i in range(len(ingr_ids[image_from_batch])):    
#         for possible_class in range(my_table.shape[0]):
#             
#             if ingr_ids[image_from_batch][i] != 0: 
#                 class_scores[possible_class] += my_table[possible_class][ingr_ids[image_from_batch][i]]
#                   
#     pred_class = np.argmax(class_scores, axis=0)   
#     
# # =============================================================================
# #     print("predicted class", pred_class)
# #     print("real class", my_classes[image_from_batch])
# #         
# # =============================================================================
#     
#     if (pred_class == my_classes[image_from_batch]):
#         accuracy += 1
#     
#     
# accuracy /= my_images.shape[0]
# print("\nAccuracy:", accuracy)
# =============================================================================
    







