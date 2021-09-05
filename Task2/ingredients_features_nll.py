# =============================================================================
# Task2a: Food101 Images Multiclass Classification using Ingredients
# Last Edit: 27/07/2021
# 
# 
# Simple Dataset
# 
# Ingredients used: 283
# Average Ingredients per image: 7.534871287128713
# Epochs 40
# Train loss: -0.420  Train acc.: 0.430  Top-5 acc.: 0.615  Test loss: -0.381  Test acc.: 0.385  Top-5 acc.: 0.606
# Training complete in 33m 45s
# Test Top-1 Accuracy = 0.4072277227722772
# Test Top-5 Accuracy = 0.6082178217821782
# 
# 
# Augmented Dataset
# 
# Ingredients used: 337
# Average Ingredients per image: 7.321413861386139
# Epochs 40
# Train loss: -0.369  Train acc.: 0.374  Top-5 acc.: 0.579  Test loss: -0.362  Test acc.: 0.365  Top-5 acc.: 0.579
# Training complete in 337m 54s
# Test Top-1 Accuracy = 0.402079207920792
# Test Top-5 Accuracy = 0.626881188118811 # Kalutero epeidi exei ekpaideutei sto na genikeuei tis foto (train_transforms)
# =============================================================================




#%% STEP 0: Import Libraries, Load necessary files and Create Dataloaders
# Import Libraries
import os
import time
import pickle
import torch
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torchsummary import summary
from InverseCooking.src.args import get_parser
from InverseCooking.src.model import get_model
from InverseCooking.src.utils.output_utils import prepare_output

# Specify data directoris for loading
image_data_dir = 'X:/datasets/food-101/images'
current_data_dir = 'X:/thesis/Task2'
im2recipe_data_dir = 'X:/thesis_outputs/InverseCooking'
output_data_dir = 'X:/thesis_outputs'

# Specify type of dataset
type_of_dataset = 'augmented_dataset'



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
img_encoder   = modules[2]
ingr_decoder  = modules[3]




#%% STEP 1: Create Dataset with Ingredients-Classes
# Create Dataloaders
def load_split_train_test(datadir, valid_size = .2, batch_size = 100):
        
    train_transforms =  transforms.Compose([
                        transforms.Resize(size=256),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
                        transforms.RandomCrop(size=224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225]) # Imagenet standards
                        ])  
    
    test_transforms =   transforms.Compose([
                        transforms.Resize(size=256),
                        transforms.CenterCrop(size=224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
                        ])
    
    
    train_data = datasets.ImageFolder(datadir, transform=train_transforms) 
    test_data = datasets.ImageFolder(datadir, transform=test_transforms)
    whole_data = datasets.ImageFolder(datadir, transform=train_transforms) # for one epoch: test transforms, for ten epochs: train transforms
    
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    
    train_idx, test_idx, whole_dataset_idx = indices[split:], indices[:split], indices
    
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    whole_sampler = SubsetRandomSampler(whole_dataset_idx)
    
    
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, num_workers=8) 
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=batch_size, num_workers=8)
    wholedatasetloader = torch.utils.data.DataLoader(whole_data, sampler=whole_sampler, batch_size=batch_size, num_workers=8)
    
    return trainloader, testloader, wholedatasetloader


trainloader, testloader, wholedatasetloader = load_split_train_test(image_data_dir, .2)



# Predict Ingredients
def label2onehot(my_list):
    my_array = np.zeros((1488,), dtype=bool)  # Type = bool for space saving
    for i in my_list:
        my_array[i] = 1     
    return my_array

# Initialize the two datasets
ingrs_per_class = np.zeros((len(classes), ingr_vocab_size))
ingrs_and_class = []

# Transfer the required modules to GPU
img_encoder, ingr_decoder = img_encoder.to(device), ingr_decoder.to(device)


# =============================================================================
# for epoch in range(10): # With 10 epochs I create an augmented dataset of 1M pairs (ingredients-classes)
# 
#     for my_images, my_classes in tqdm(wholedatasetloader):
#         
#         my_images = my_images.to(device)
#         my_classes = my_classes.to(device)
#           
#         with torch.no_grad():
#             
#             features = img_encoder(my_images)    
#             ingr_ids, ingr_logits = ingr_decoder.sample(None, None, greedy=True, temperature=1.0, img_features=features, first_token_value=0, replacement=False)     
#             
#             ## Measure ingredients for each class
#             for i in range(len(ingr_ids)):
#                 
#                 ## Create Dataset          
#                 tempDict = {'ingredients': label2onehot(ingr_ids[i].tolist()), 
#                             'class': my_classes[i].item()}
#                 
#                 ingrs_and_class.append(tempDict)
#                             
#                 for j in range(len(ingr_ids[i])):
#                                 
#                     ingrs_per_class[my_classes[i]][ingr_ids[i][j]] += 1
#     
#         torch.cuda.empty_cache()
# 
# 
# 
# # Save the created datasets
# with open(os.path.join(output_data_dir, 'Task2/' + type_of_dataset + '/ingrs_and_class.pkl'), 'wb') as fp:   
#     pickle.dump(ingrs_and_class, fp)
# 
# with open(os.path.join(output_data_dir, 'Task2/' + type_of_dataset + '/ingrs_per_class.pkl'), 'wb') as fp:   
#     pickle.dump(ingrs_per_class, fp)
# =============================================================================





#%% STEP 2: LOAD ingrs_and_classes and ingrs_per_class datasets
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






#%% STEP 3: Train a Simple Classifier

class ingr2classClassifier(nn.Module):
    
    def __init__(self):
        super(ingr2classClassifier, self).__init__()          
        self.fc1 = nn.Linear(in_features=1488, out_features=101, bias=True)
        self.softmax = nn.Softmax(dim=1)  

    def forward(self, x):      
        x = self.fc1(x)     
        x = self.softmax(x).view(-1, 101)
        return x


task2_Classifier = ingr2classClassifier().to(device)



# Train loop Function
def training_loop(epochs, criterion, optimizer, scheduler, train_loader, val_loader, train_losses, train_accuracies, train_top5_accuracies, test_losses, test_accuracies, test_top5_accuracies):

    print(f"\n Training for {epochs} epochs  ...")    
    time.sleep(0.2)

    for epoch in range(epochs):
        
        print(f"\n  ## Epoch {epoch+1}/{epochs} with learning rate = {optimizer.param_groups[0]['lr']} ##\n")
        time.sleep(0.2)
        
        # Training Process
        task2_Classifier.train()
        train_loss, train_accuracy, train_top5_accuracy = 0, 0, 0
        
        for ingredients_batch, class_batch in tqdm(train_loader):
            # Loading next batch
            ingredients_batch, class_batch = ingredients_batch.float().to(device), class_batch.to(device)
            # Initialize Optimizer
            optimizer.zero_grad()
            # Forward pass: compute predicted outputs
            logps = task2_Classifier.forward(ingredients_batch)
            # Calculating Loss 
            loss = criterion(logps, class_batch)
            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # Perform a single optimization step (parameter update)
            optimizer.step()
            # Summing Loss
            train_loss += loss.item()
            # Calculating Accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == class_batch.view(*top_class.shape)
            train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            # Top 5 Accuracy
            top5_p, top5_class = ps.topk(5)
            for i in range(top5_class.shape[0]):
                for j in range(top5_class.shape[1]):
                    if top5_class[i][j] == class_batch[i]:
                        train_top5_accuracy += 1
                        break
            
            
        # Evaluation Process
        task2_Classifier.eval()
        test_loss, test_accuracy, test_top5_accuracy = 0, 0, 0
        
        with torch.no_grad():
            for ingredients_batch, class_batch in tqdm(val_loader):
                # Loading next batch
                ingredients_batch, class_batch = ingredients_batch.float().to(device), class_batch.to(device)
                # Forward Pass: Predicting class
                logps = task2_Classifier.forward(ingredients_batch)
                # Calculating Loss 
                batch_loss = criterion(logps, class_batch)
                # Summing Loss
                test_loss += batch_loss.item()
                # Calculating Accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == class_batch.view(*top_class.shape)
                test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                # Top 5 Accuracy
                top5_p, top5_class = ps.topk(5)
                for i in range(top5_class.shape[0]):
                    for j in range(top5_class.shape[1]):
                        if top5_class[i][j] == class_batch[i]:
                            test_top5_accuracy += 1
                            break
    
        # Updating lists  
        train_losses.append(train_loss/len(train_loader)) # Missing last batch, which is not exaclty 64
        train_accuracies.append(train_accuracy/len(train_loader)) 
        train_top5_accuracies.append(train_top5_accuracy/(len(train_loader)*batch_size)) 
        
        test_losses.append(test_loss/len(val_loader))   
        test_accuracies.append(test_accuracy/len(val_loader)) 
        test_top5_accuracies.append(test_top5_accuracy/(len(val_loader)*batch_size)) 
                    
        print(f"\nTrain loss: {train_loss/len(train_loader):.3f}  "
              f"Train acc.: {train_accuracy/len(train_loader):.3f}  "
              f"Top-5 acc.: {train_top5_accuracy/(len(train_loader)*batch_size):.3f}  "
              f"Test loss: {test_loss/len(val_loader):.3f}  "
              f"Test acc.: {test_accuracy/len(val_loader):.3f}  "
              f"Top-5 acc.: {test_top5_accuracy/(len(val_loader)*batch_size):.3f}")
          
        # Learning rate scheduler step
        scheduler.step(test_accuracy / len(val_loader)) 
        time.sleep(0.5)
  


# Training Parameters
epochs = 40
criterion = nn.NLLLoss()
optimizer = optim.Adam(task2_Classifier.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', threshold=0.006, factor=0.2, patience=4)
train_losses, train_accuracies, train_top5_accuracies, test_losses, test_accuracies, test_top5_accuracies = [], [], [], [], [], []

# Training Process
since = time.time()  
training_loop(epochs, criterion, optimizer, scheduler, train_loader, val_loader, train_losses, train_accuracies, train_top5_accuracies, test_losses, test_accuracies, test_top5_accuracies)                   
time_elapsed = time.time() - since
print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))   
 

# Saving trained weights        
torch.save(task2_Classifier, os.path.join(output_data_dir, 'Task2/' + type_of_dataset + '/model.pth'))

# Plot metrics
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.savefig(os.path.join(output_data_dir, 'Task2/' + type_of_dataset + '/losses.jpg'), transparent=True, dpi=200)
plt.show()

plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.legend(frameon=False)
plt.savefig(os.path.join(output_data_dir, 'Task2/' + type_of_dataset + '/accuracies.jpg'), transparent=True, dpi=200)
plt.show()

plt.plot(train_top5_accuracies, label='Train Top-5 Accuracy')
plt.plot(test_top5_accuracies, label='Test Top-5 Accuracy')
plt.legend(frameon=False)
plt.savefig(os.path.join(output_data_dir, 'Task2/' + type_of_dataset + '/top5_accuracies.jpg'), transparent=True, dpi=200)
plt.show()






#%% STEP 4: Evaluation

# Load model
# task2_Classifier.load_state_dict(torch.load(os.path.join(current_data_dir, 'output/task2_model.pth')))
model = torch.load(os.path.join(output_data_dir, 'Task2/' + type_of_dataset + '/model.pth'))
model = model.to(device)

def label2onehotRemastered(myTensor):
    my_array = np.zeros((myTensor.shape[0], 1488), dtype=int)  
    for i in range(myTensor.shape[0]): 
        for j in range(myTensor.shape[1]):
            my_array[i][myTensor[i][j]] = 1          
    return my_array


# Transfer the required modules to GPU
img_encoder, ingr_decoder = img_encoder.to(device), ingr_decoder.to(device)

# Evaluation
top1_acc, top5_acc = 0, 0

for my_images, my_classes in tqdm(testloader):
    
    my_images, my_classes = my_images.to(device), my_classes.to(device)
    
    with torch.no_grad():
        
        features = img_encoder(my_images)
        ingr_ids, ingr_logits = ingr_decoder.sample(None, None, greedy=True, temperature=1.0, img_features=features, first_token_value=0, replacement=False)
        
        onehotIngredients = label2onehotRemastered(ingr_ids)
        onehotIngredients = torch.from_numpy(onehotIngredients).to(device).float()
        predictions = model(onehotIngredients)
    
        # Top-1 Accuracy
        for i in range(len(predictions)):
            if predictions[i].argmax() == my_classes[i]:
                top1_acc += 1
    
        # Top-5 Accuracy
        top5_classes = predictions.topk(5)
        for i in range(top5_classes.indices.shape[0]):
            for j in range(top5_classes.indices.shape[1]):       
                if top5_classes.indices[i][j] == my_classes[i]:
                    top5_acc += 1
                    break

    torch.cuda.empty_cache()


print("\n\n")
print("="*40)
print("Test Top-1 Accuracy =", top1_acc/(len(testloader) * my_images.shape[0]))
print("Test Top-5 Accuracy =", top5_acc/(len(testloader) * my_images.shape[0]))
print("="*40)
print("\n\n")


