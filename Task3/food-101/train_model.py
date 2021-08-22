# =============================================================================
# Script to train and evaluate an LSTM model for the classification problem of Food-101
# Reloaded Edition of 2 LSTMS of 256 and 512 neurons
# Author: Dimitrios-Marios Exarchou
# Last modified: 21/8/2021   
# 
# Top-1   Accuracy = 34.49 %
# Top-10  Accuracy = 53.56 %
# Top-20  Accuracy = 61.42 %
# Top-50  Accuracy = 64.86 %
#
# Training time: 46m 54s
# =============================================================================


#%% STEP 1: Libraries
import os
import time
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras import metrics
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Dropout, Bidirectional
from keras.initializers import Constant
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from tqdm.keras import TqdmCallback
from gensim.models import KeyedVectors
from gensim.similarities.annoy import AnnoyIndexer
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight




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




#%% STEP 3: Building LSTM model
LSTM_model = Sequential()
LSTM_model.add(Embedding(len(ingr_vocab),                                   # number of unique tokens
                         EMBEDDING_VECTOR_LENGTH,                           # length of embedded vectors
                         embeddings_initializer=Constant(embedding_matrix), # initialization of matrix 
                         input_length=MAX_SEQUENCE_LENGTH,                  # max number of ingredients
                         trainable=True))
LSTM_model.add(Dropout(0.2))
LSTM_model.add(Bidirectional(LSTM(512, dropout=0.2, recurrent_dropout=0.5, return_sequences=True)))
LSTM_model.add(Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.5)))
LSTM_model.add(Dense(EMBEDDING_VECTOR_LENGTH, activation='relu'))

LSTM_model.summary()




#%% STEP 4: Training Process

# Function to get Learning Rate
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

# Learning Rate schedule
def lr_step_decay(epoch):
	initial_lrate = 0.005
	drop = 0.2
	epochs_drop = 10
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate


# Optimizer
optimizer = Adam()
lr_metric = get_lr_metric(optimizer)


# Model Compile
LSTM_model.compile(loss='mse', optimizer=optimizer, metrics=[metrics.CosineSimilarity(axis=1)])



# Model Train
since = time.time()  
history = LSTM_model.fit(X_train, y_train, epochs=24, batch_size=256, validation_data=(X_valid, y_valid), verbose=1, callbacks=[LearningRateScheduler(lr_step_decay, verbose=1)]) 
time_elapsed = time.time() - since
print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))   
 


# =============================================================================
# Training log stats 
# 
# Epoch 00001: LearningRateScheduler reducing learning rate to 0.005.
# Epoch 1/24
# 277/277 [==============================] - 115s 417ms/step - loss: 0.0336 - cosine_similarity: 0.4431 - val_loss: 0.0323 - val_cosine_similarity: 0.4776
# 
# Epoch 00002: LearningRateScheduler reducing learning rate to 0.005.
# Epoch 2/24
# 277/277 [==============================] - 115s 414ms/step - loss: 0.0314 - cosine_similarity: 0.4962 - val_loss: 0.0309 - val_cosine_similarity: 0.5079
# 
# Epoch 00003: LearningRateScheduler reducing learning rate to 0.005.
# Epoch 3/24
# 277/277 [==============================] - 115s 414ms/step - loss: 0.0307 - cosine_similarity: 0.5135 - val_loss: 0.0305 - val_cosine_similarity: 0.5175
# 
# Epoch 00004: LearningRateScheduler reducing learning rate to 0.005.
# Epoch 4/24
# 277/277 [==============================] - 115s 416ms/step - loss: 0.0303 - cosine_similarity: 0.5222 - val_loss: 0.0302 - val_cosine_similarity: 0.5235
# 
# Epoch 00005: LearningRateScheduler reducing learning rate to 0.005.
# Epoch 5/24
# 277/277 [==============================] - 117s 423ms/step - loss: 0.0301 - cosine_similarity: 0.5268 - val_loss: 0.0302 - val_cosine_similarity: 0.5251
# 
# Epoch 00006: LearningRateScheduler reducing learning rate to 0.005.
# Epoch 6/24
# 277/277 [==============================] - 116s 417ms/step - loss: 0.0299 - cosine_similarity: 0.5301 - val_loss: 0.0301 - val_cosine_similarity: 0.5270
# 
# Epoch 00007: LearningRateScheduler reducing learning rate to 0.005.
# Epoch 7/24
# 277/277 [==============================] - 118s 428ms/step - loss: 0.0298 - cosine_similarity: 0.5326 - val_loss: 0.0300 - val_cosine_similarity: 0.5287
# 
# Epoch 00008: LearningRateScheduler reducing learning rate to 0.005.
# Epoch 8/24
# 277/277 [==============================] - 119s 429ms/step - loss: 0.0297 - cosine_similarity: 0.5349 - val_loss: 0.0300 - val_cosine_similarity: 0.5286
# 
# Epoch 00009: LearningRateScheduler reducing learning rate to 0.005.
# Epoch 9/24
# 277/277 [==============================] - 117s 421ms/step - loss: 0.0296 - cosine_similarity: 0.5367 - val_loss: 0.0300 - val_cosine_similarity: 0.5283
# 
# Epoch 00010: LearningRateScheduler reducing learning rate to 0.001.
# Epoch 10/24
# 277/277 [==============================] - 118s 425ms/step - loss: 0.0292 - cosine_similarity: 0.5445 - val_loss: 0.0299 - val_cosine_similarity: 0.5310
# 
# Epoch 00011: LearningRateScheduler reducing learning rate to 0.001.
# Epoch 11/24
# 277/277 [==============================] - 121s 436ms/step - loss: 0.0290 - cosine_similarity: 0.5473 - val_loss: 0.0299 - val_cosine_similarity: 0.5309
# 
# Epoch 00012: LearningRateScheduler reducing learning rate to 0.001.
# Epoch 12/24
# 277/277 [==============================] - 116s 418ms/step - loss: 0.0289 - cosine_similarity: 0.5492 - val_loss: 0.0299 - val_cosine_similarity: 0.5305
# 
# Epoch 00013: LearningRateScheduler reducing learning rate to 0.001.
# Epoch 13/24
# 277/277 [==============================] - 115s 416ms/step - loss: 0.0289 - cosine_similarity: 0.5506 - val_loss: 0.0300 - val_cosine_similarity: 0.5294
# 
# Epoch 00014: LearningRateScheduler reducing learning rate to 0.001.
# Epoch 14/24
# 277/277 [==============================] - 115s 415ms/step - loss: 0.0288 - cosine_similarity: 0.5522 - val_loss: 0.0300 - val_cosine_similarity: 0.5296
# 
# Epoch 00015: LearningRateScheduler reducing learning rate to 0.001.
# Epoch 15/24
# 277/277 [==============================] - 115s 414ms/step - loss: 0.0287 - cosine_similarity: 0.5539 - val_loss: 0.0301 - val_cosine_similarity: 0.5284
# 
# Epoch 00016: LearningRateScheduler reducing learning rate to 0.001.
# Epoch 16/24
# 277/277 [==============================] - 115s 414ms/step - loss: 0.0286 - cosine_similarity: 0.5553 - val_loss: 0.0301 - val_cosine_similarity: 0.5276
# 
# Epoch 00017: LearningRateScheduler reducing learning rate to 0.001.
# Epoch 17/24
# 277/277 [==============================] - 115s 414ms/step - loss: 0.0285 - cosine_similarity: 0.5569 - val_loss: 0.0302 - val_cosine_similarity: 0.5273
# 
# Epoch 00018: LearningRateScheduler reducing learning rate to 0.001.
# Epoch 18/24
# 277/277 [==============================] - 115s 414ms/step - loss: 0.0285 - cosine_similarity: 0.5588 - val_loss: 0.0301 - val_cosine_similarity: 0.5271
# 
# Epoch 00019: LearningRateScheduler reducing learning rate to 0.001.
# Epoch 19/24
# 277/277 [==============================] - 115s 415ms/step - loss: 0.0284 - cosine_similarity: 0.5604 - val_loss: 0.0302 - val_cosine_similarity: 0.5261
# 
# Epoch 00020: LearningRateScheduler reducing learning rate to 0.00020000000000000004.
# Epoch 20/24
# 277/277 [==============================] - 115s 414ms/step - loss: 0.0281 - cosine_similarity: 0.5649 - val_loss: 0.0303 - val_cosine_similarity: 0.5254
# 
# Epoch 00021: LearningRateScheduler reducing learning rate to 0.00020000000000000004.
# Epoch 21/24
# 277/277 [==============================] - 115s 416ms/step - loss: 0.0281 - cosine_similarity: 0.5661 - val_loss: 0.0303 - val_cosine_similarity: 0.5252
# 
# Epoch 00022: LearningRateScheduler reducing learning rate to 0.00020000000000000004.
# Epoch 22/24
# 277/277 [==============================] - 118s 428ms/step - loss: 0.0280 - cosine_similarity: 0.5667 - val_loss: 0.0303 - val_cosine_similarity: 0.5249
# 
# Epoch 00023: LearningRateScheduler reducing learning rate to 0.00020000000000000004.
# Epoch 23/24
# 277/277 [==============================] - 120s 435ms/step - loss: 0.0280 - cosine_similarity: 0.5675 - val_loss: 0.0303 - val_cosine_similarity: 0.5246
# 
# Epoch 00024: LearningRateScheduler reducing learning rate to 0.00020000000000000004.
# Epoch 24/24
# 277/277 [==============================] - 121s 438ms/step - loss: 0.0280 - cosine_similarity: 0.5681 - val_loss: 0.0304 - val_cosine_similarity: 0.5241
# 
# Training complete in 46m 54s
# =============================================================================



#%% STEP 5: Evaluation with Annoy Indexer

# Find the accuracies of classes
classes_acc = dict() 

for i in range(len(y_test_int)):
    classes_acc[classes_reloaded[y_test_int[i]]] = 0

# Find Accuracies 
top1_accuracy, top10_accuracy, top20_accuracy, top50_accuracy = 0, 0, 0, 0

# Predictions
predictions = LSTM_model.predict(X_test, batch_size=256, verbose=1)
y_pred = []

for i in tqdm(range(len(predictions))):

    original_class = y_test_int[i]
    # Find the approximate neighbors with Annoy
    approximate_neighbors = word2vec_model.most_similar([predictions[i]], topn=1000, indexer=annoy_index)
    close_words = []
    for neighbor in approximate_neighbors:
        close_words.append(neighbor[0])

    # Find the classes that are involved in approximate neighbors
    ranked_classes = []
    for potential_class in close_words:
        if potential_class in classes_reloaded:
            ranked_classes.append(potential_class)
    
    if ranked_classes != []:
        y_pred.append(ranked_classes[0])
    else:
        y_pred.append(0)
    
    # Top-1 Accuracy
    if classes_reloaded[original_class] in ranked_classes[:1]:
        top1_accuracy += 1
        classes_acc[classes_reloaded[original_class]] += 1
        
    # Top-10 Accuracy
    if classes_reloaded[original_class] in ranked_classes[:10]:
        top10_accuracy += 1
                
    # Top-20 Accuracy
    if classes_reloaded[original_class] in ranked_classes[:20]:
        top20_accuracy += 1
 
    # Top-50 Accuracy
    if classes_reloaded[original_class] in ranked_classes[:50]:
        top50_accuracy += 1

            
print()
print('=' * 30)
print(f'Top-1   Accuracy = {  top1_accuracy * 100 / len(X_test) :.2f} %')
print(f'Top-10  Accuracy = { top10_accuracy * 100 / len(X_test) :.2f} %')
print(f'Top-20  Accuracy = { top20_accuracy * 100 / len(X_test) :.2f} %')
print(f'Top-50  Accuracy = { top50_accuracy * 100 / len(X_test) :.2f} %')
print('=' * 30)





#%% STEP 6: Find classes' Accuracy & Confusion Matrix
classes_distribution_test = dict()

for i in range(len(y_test_int)):
    classes_distribution_test[classes_reloaded[y_test_int[i]]] = 0

for i in range(len(y_test_int)):
    classes_distribution_test[classes_reloaded[y_test_int[i]]] += 1


counter = 0

for key in classes_acc:
    classes_acc[key] /= classes_distribution_test[key]
    if classes_acc[key] != 0:
        counter += 1
        
print('Number of classes with at least one correct prediction:', counter) #257


# Confusion Matrix
y_true = []

for i in range(len(y_test_int)):
    y_true.append(classes_reloaded[y_test_int[i]])
    

classes_names = []
for key in classes_reloaded:
    if key not in classes_names:
        classes_names.append(key)


cm = confusion_matrix(y_true, y_pred, labels=classes_names)

# Measure top-1 Accuracy from the diagonal of confusion matrix
accuracy = 0

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        if i == j:
            accuracy += cm[i][j]

accuracy /= np.sum(cm)
accuracy *= 100

print(f'Top-1   Accuracy = {  accuracy:.2f} %')


# Transform confusion matrix into dataframe
cm_df = pd.DataFrame(cm, columns=classes_names, index=classes_names)





#%% Save classes stats, confusion matrix and model

with open(os.path.join(save_output_dir, 'classes_acc.pkl'), 'wb') as fp:   
    pickle.dump(classes_acc, fp)

with open(os.path.join(save_output_dir, 'classes_distribution_test.pkl'), 'wb') as fp:    
    pickle.dump(classes_distribution_test, fp)


with open(os.path.join(save_output_dir, 'confusion_matrix.pkl'), 'wb') as fp:    
    pickle.dump(cm, fp)

with open(os.path.join(save_output_dir, 'confusion_matrix_dataframe.pkl'), 'wb') as fp:     
    pickle.dump(cm_df, fp)


with open(os.path.join(save_output_dir, 'history.pkl'), 'wb') as fp:
    pickle.dump(history.history, fp)

LSTM_model.save(os.path.join(save_output_dir, 'LSTM_model_34.49.h5'))






#%% Plot Learning curves

model = load_model(os.path.join(save_output_dir, 'LSTM_model_34.49.h5'))

with open(os.path.join(save_output_dir, 'history.pkl'), 'rb') as fp:
    train_history = pickle.load(fp)

loss_list        = train_history['loss']
cos_sim_list     = train_history['cosine_similarity']
val_loss_list    = train_history['val_loss']
val_cos_sim_list = train_history['val_cosine_similarity']



plt.plot(loss_list, label='Training loss')
plt.plot(val_loss_list, label='Validation loss')
plt.legend(frameon=False)
plt.savefig(os.path.join(save_output_dir, 'loss.jpg'), transparent=True, dpi=200)
plt.show()

plt.plot(cos_sim_list, label='Train Cos. Sim.')
plt.plot(val_cos_sim_list, label='Test Cos. Sim.')
plt.legend(frameon=False)
plt.savefig(os.path.join(save_output_dir, 'cos_sim.jpg'), transparent=True, dpi=200)
plt.show()

