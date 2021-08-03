# Script for the visualization of ResNet-50 kernels and their effects

# Libraries
import os
import cv2 as cv
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models, transforms


# Load the model
model = models.resnet50(pretrained=True)
print(model)
model_weights = [] # we will save the conv layer weights in this list
conv_layers = [] # we will save the 49 conv layers in this list
# Get all the model children as list
model_children = list(model.children())


# Define current directory
cur_dir = 'X:/thesis/Task1'


# Counter to keep count of the convolutional layers
counter = 0 
# append all the conv layers and their respective weights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolutional layers: {counter}")


# Take a look at the conv layers and the respective weights
for weight, conv in zip(model_weights, conv_layers):
    # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
    print(f"CONV: {conv} ====> SHAPE: {weight.shape}")
    
     
# Visualize the first convolutional layer's filters
plt.figure(figsize=(20, 17))
for i, filter in enumerate(model_weights[0]):
    plt.subplot(8, 8, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
    plt.imshow(filter[0, :, :].detach(), cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(cur_dir, 'vis_kernels/filter.png'))
plt.show()


# Read and visualize an image
img = cv.imread(os.path.join(cur_dir, 'vis_kernels/burger.jpg'))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.show()
# define the transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
img = np.array(img)
# Apply the transforms
img = transform(img)
print(img.size())
# Unsqueeze to add a batch dimension
img = img.unsqueeze(0)
print(img.size())


# Pass the image through all the layers
results = [conv_layers[0](img)]

for i in range(1, len(conv_layers)):
    # pass the result from the last layer to the next layer
    results.append(conv_layers[i](results[-1]))
# make a copy of the `results`
outputs = results

print(img.shape)
step = conv_layers[0](img)
print(step.shape)
step = conv_layers[1](step)
print(step.shape)


# Visualize 64 features from each layer 
# (although there are more feature maps in the upper layers)
for num_layer in range(len(outputs)):
    plt.figure(figsize=(30, 30))
    layer_viz = outputs[num_layer][0, :, :, :]
    layer_viz = layer_viz.data
    print(layer_viz.size())
    for i, filter in enumerate(layer_viz):
        if i == 64: # we will visualize only 8x8 blocks from each layer
            break
        plt.subplot(8, 8, i + 1)
        plt.imshow(filter, cmap='gray')
        plt.axis("off")
    print(f"Saving layer {num_layer} feature maps...")
    plt.savefig(os.path.join(cur_dir, f'vis_kernels/layer_{num_layer}.png'))
    # plt.show()
    plt.close()


