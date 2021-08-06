# =============================================================================
# Script 5: to train and evaluate an LSTM model for the classification problem
# Edition: bidirectional LSTM with trainable embedding matrix and replaced noisy classes II
# Author: Dimitrios-Marios Exarchou
# Last modified: 6/8/2021   
# 
# Top-1   Accuracy = 30.61 %
# Top-10  Accuracy = 43.53 %
# Top-20  Accuracy = 48.22 %
# Top-50  Accuracy = 55.85 %
# 
# Evaluation time: 13:32
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
current_data_dir   = 'X:/thesis/Task3'
output_data_dir    = 'X:/thesis_past/data'                # I have to copy it to 'X:/thesis_outputs/Task3'
image_data_dir     = 'X:/thesis_past/data/images/'        # I have to copy it to 'X:/datasets/recipe1m/images'
im2recipe_data_dir = 'X:/thesis_outputs/InverseCooking'
word2vec_data      = 'X:/thesis_outputs/Task3/word2vec_data'
output_dataset_dir = 'X:/thesis_outputs/Task3/final_dataset'
annoy_indexer_dir  = 'X:/thesis_outputs/Task3/annoy_index'
save_output_dir    = 'X:/thesis_outputs/Task3/5'

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




#%% STEP 3: Building LSTM model
LSTM_model = Sequential()
LSTM_model.add(Embedding(len(ingr_vocab),                                   # number of unique tokens
                         EMBEDDING_VECTOR_LENGTH,                           # length of embedded vectors
                         embeddings_initializer=Constant(embedding_matrix), # initialization of matrix 
                         input_length=MAX_SEQUENCE_LENGTH,                  # max number of ingredients
                         trainable=True))
LSTM_model.add(Dropout(0.2))
LSTM_model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.5, return_sequences=True)))
LSTM_model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.5)))
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
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate


# Optimizer
optimizer = Adam()
lr_metric = get_lr_metric(optimizer)


# Model Compile
LSTM_model.compile(loss='mse', optimizer=optimizer, metrics=[metrics.CosineSimilarity(axis=1)])



# Model Train
since = time.time()  
history = LSTM_model.fit(X_train, y_train, epochs=30, batch_size=1024, validation_data=(X_valid, y_valid), verbose=1, callbacks=[LearningRateScheduler(lr_step_decay, verbose=1)]) 
time_elapsed = time.time() - since
print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))   
 


# =============================================================================
# Training log stats 
# 
# Epoch 00001: LearningRateScheduler reducing learning rate to 0.005.
# Epoch 1/30
# 272/272 [==============================] - 110s 406ms/step - loss: 0.0294 - cosine_similarity: 0.4843 - lr: 0.0050 - val_loss: 0.0287 - val_cosine_similarity: 0.5042 - val_lr: 0.0050
# 
# Epoch 00002: LearningRateScheduler reducing learning rate to 0.005.
# Epoch 2/30
# 272/272 [==============================] - 108s 396ms/step - loss: 0.0284 - cosine_similarity: 0.5106 - lr: 0.0050 - val_loss: 0.0281 - val_cosine_similarity: 0.5174 - val_lr: 0.0050
# 
# Epoch 00003: LearningRateScheduler reducing learning rate to 0.005.
# Epoch 3/30
# 272/272 [==============================] - 107s 392ms/step - loss: 0.0280 - cosine_similarity: 0.5198 - lr: 0.0050 - val_loss: 0.0279 - val_cosine_similarity: 0.5230 - val_lr: 0.0050
# 
# Epoch 00004: LearningRateScheduler reducing learning rate to 0.005.
# Epoch 4/30
# 272/272 [==============================] - 107s 394ms/step - loss: 0.0278 - cosine_similarity: 0.5241 - lr: 0.0050 - val_loss: 0.0277 - val_cosine_similarity: 0.5257 - val_lr: 0.0050
# 
# Epoch 00005: LearningRateScheduler reducing learning rate to 0.005.
# Epoch 5/30
# 272/272 [==============================] - 108s 397ms/step - loss: 0.0277 - cosine_similarity: 0.5269 - lr: 0.0050 - val_loss: 0.0277 - val_cosine_similarity: 0.5277 - val_lr: 0.0050
# 
# Epoch 00006: LearningRateScheduler reducing learning rate to 0.005.
# Epoch 6/30
# 272/272 [==============================] - 108s 397ms/step - loss: 0.0276 - cosine_similarity: 0.5287 - lr: 0.0050 - val_loss: 0.0276 - val_cosine_similarity: 0.5288 - val_lr: 0.0050
# 
# Epoch 00007: LearningRateScheduler reducing learning rate to 0.005.
# Epoch 7/30
# 272/272 [==============================] - 110s 406ms/step - loss: 0.0276 - cosine_similarity: 0.5298 - lr: 0.0050 - val_loss: 0.0276 - val_cosine_similarity: 0.5298 - val_lr: 0.0050
# 
# Epoch 00008: LearningRateScheduler reducing learning rate to 0.005.
# Epoch 8/30
# 272/272 [==============================] - 109s 400ms/step - loss: 0.0275 - cosine_similarity: 0.5308 - lr: 0.0050 - val_loss: 0.0275 - val_cosine_similarity: 0.5303 - val_lr: 0.0050
# 
# Epoch 00009: LearningRateScheduler reducing learning rate to 0.005.
# Epoch 9/30
# 272/272 [==============================] - 109s 400ms/step - loss: 0.0275 - cosine_similarity: 0.5314 - lr: 0.0050 - val_loss: 0.0275 - val_cosine_similarity: 0.5305 - val_lr: 0.0050
# 
# Epoch 00010: LearningRateScheduler reducing learning rate to 0.001.
# Epoch 10/30
# 272/272 [==============================] - 108s 397ms/step - loss: 0.0274 - cosine_similarity: 0.5337 - lr: 0.0010 - val_loss: 0.0274 - val_cosine_similarity: 0.5321 - val_lr: 1.0000e-03
# 
# Epoch 00011: LearningRateScheduler reducing learning rate to 0.001.
# Epoch 11/30
# 272/272 [==============================] - 108s 398ms/step - loss: 0.0273 - cosine_similarity: 0.5341 - lr: 0.0010 - val_loss: 0.0274 - val_cosine_similarity: 0.5323 - val_lr: 1.0000e-03
# 
# Epoch 00012: LearningRateScheduler reducing learning rate to 0.001.
# Epoch 12/30
# 272/272 [==============================] - 109s 400ms/step - loss: 0.0273 - cosine_similarity: 0.5345 - lr: 0.0010 - val_loss: 0.0274 - val_cosine_similarity: 0.5324 - val_lr: 1.0000e-03
# 
# Epoch 00013: LearningRateScheduler reducing learning rate to 0.001.
# Epoch 13/30
# 272/272 [==============================] - 109s 400ms/step - loss: 0.0273 - cosine_similarity: 0.5347 - lr: 0.0010 - val_loss: 0.0274 - val_cosine_similarity: 0.5325 - val_lr: 1.0000e-03
# 
# Epoch 00014: LearningRateScheduler reducing learning rate to 0.001.
# Epoch 14/30
# 272/272 [==============================] - 109s 402ms/step - loss: 0.0273 - cosine_similarity: 0.5349 - lr: 0.0010 - val_loss: 0.0274 - val_cosine_similarity: 0.5325 - val_lr: 1.0000e-03
# 
# Epoch 00015: LearningRateScheduler reducing learning rate to 0.001.
# Epoch 15/30
# 272/272 [==============================] - 107s 395ms/step - loss: 0.0273 - cosine_similarity: 0.5351 - lr: 0.0010 - val_loss: 0.0274 - val_cosine_similarity: 0.5325 - val_lr: 1.0000e-03
# 
# Epoch 00016: LearningRateScheduler reducing learning rate to 0.001.
# Epoch 16/30
# 272/272 [==============================] - 107s 394ms/step - loss: 0.0273 - cosine_similarity: 0.5353 - lr: 0.0010 - val_loss: 0.0274 - val_cosine_similarity: 0.5327 - val_lr: 1.0000e-03
# 
# Epoch 00017: LearningRateScheduler reducing learning rate to 0.001.
# Epoch 17/30
# 272/272 [==============================] - 108s 396ms/step - loss: 0.0273 - cosine_similarity: 0.5354 - lr: 0.0010 - val_loss: 0.0274 - val_cosine_similarity: 0.5326 - val_lr: 1.0000e-03
# 
# Epoch 00018: LearningRateScheduler reducing learning rate to 0.001.
# Epoch 18/30
# 272/272 [==============================] - 107s 392ms/step - loss: 0.0273 - cosine_similarity: 0.5356 - lr: 0.0010 - val_loss: 0.0274 - val_cosine_similarity: 0.5328 - val_lr: 1.0000e-03
# 
# Epoch 00019: LearningRateScheduler reducing learning rate to 0.001.
# Epoch 19/30
# 272/272 [==============================] - 108s 396ms/step - loss: 0.0273 - cosine_similarity: 0.5358 - lr: 0.0010 - val_loss: 0.0274 - val_cosine_similarity: 0.5328 - val_lr: 1.0000e-03
# 
# Epoch 00020: LearningRateScheduler reducing learning rate to 0.00020000000000000004.
# Epoch 20/30
# 272/272 [==============================] - 108s 399ms/step - loss: 0.0272 - cosine_similarity: 0.5365 - lr: 2.0000e-04 - val_loss: 0.0274 - val_cosine_similarity: 0.5331 - val_lr: 2.0000e-04
# 
# Epoch 00021: LearningRateScheduler reducing learning rate to 0.00020000000000000004.
# Epoch 21/30
# 272/272 [==============================] - 108s 397ms/step - loss: 0.0272 - cosine_similarity: 0.5367 - lr: 2.0000e-04 - val_loss: 0.0274 - val_cosine_similarity: 0.5330 - val_lr: 2.0000e-04
# 
# Epoch 00022: LearningRateScheduler reducing learning rate to 0.00020000000000000004.
# Epoch 22/30
# 272/272 [==============================] - 108s 398ms/step - loss: 0.0272 - cosine_similarity: 0.5367 - lr: 2.0000e-04 - val_loss: 0.0274 - val_cosine_similarity: 0.5332 - val_lr: 2.0000e-04
# 
# Epoch 00023: LearningRateScheduler reducing learning rate to 0.00020000000000000004.
# Epoch 23/30
# 272/272 [==============================] - 109s 399ms/step - loss: 0.0272 - cosine_similarity: 0.5368 - lr: 2.0000e-04 - val_loss: 0.0274 - val_cosine_similarity: 0.5331 - val_lr: 2.0000e-04
# 
# Epoch 00024: LearningRateScheduler reducing learning rate to 0.00020000000000000004.
# Epoch 24/30
# 272/272 [==============================] - 107s 395ms/step - loss: 0.0272 - cosine_similarity: 0.5368 - lr: 2.0000e-04 - val_loss: 0.0274 - val_cosine_similarity: 0.5331 - val_lr: 2.0000e-04
# 
# Epoch 00025: LearningRateScheduler reducing learning rate to 0.00020000000000000004.
# Epoch 25/30
# 272/272 [==============================] - 107s 394ms/step - loss: 0.0272 - cosine_similarity: 0.5368 - lr: 2.0000e-04 - val_loss: 0.0274 - val_cosine_similarity: 0.5331 - val_lr: 2.0000e-04
# 
# Epoch 00026: LearningRateScheduler reducing learning rate to 0.00020000000000000004.
# Epoch 26/30
# 272/272 [==============================] - 109s 400ms/step - loss: 0.0272 - cosine_similarity: 0.5369 - lr: 2.0000e-04 - val_loss: 0.0274 - val_cosine_similarity: 0.5330 - val_lr: 2.0000e-04
# 
# Epoch 00027: LearningRateScheduler reducing learning rate to 0.00020000000000000004.
# Epoch 27/30
# 272/272 [==============================] - 109s 399ms/step - loss: 0.0272 - cosine_similarity: 0.5370 - lr: 2.0000e-04 - val_loss: 0.0274 - val_cosine_similarity: 0.5331 - val_lr: 2.0000e-04
# 
# Epoch 00028: LearningRateScheduler reducing learning rate to 0.00020000000000000004.
# Epoch 28/30
# 272/272 [==============================] - 109s 400ms/step - loss: 0.0272 - cosine_similarity: 0.5370 - lr: 2.0000e-04 - val_loss: 0.0274 - val_cosine_similarity: 0.5330 - val_lr: 2.0000e-04
# 
# Epoch 00029: LearningRateScheduler reducing learning rate to 0.00020000000000000004.
# Epoch 29/30
# 272/272 [==============================] - 109s 400ms/step - loss: 0.0272 - cosine_similarity: 0.5371 - lr: 2.0000e-04 - val_loss: 0.0274 - val_cosine_similarity: 0.5331 - val_lr: 2.0000e-04
# 
# Epoch 00030: LearningRateScheduler reducing learning rate to 4.000000000000001e-05.
# Epoch 30/30
# 272/272 [==============================] - 109s 400ms/step - loss: 0.0272 - cosine_similarity: 0.5372 - lr: 4.0000e-05 - val_loss: 0.0274 - val_cosine_similarity: 0.5332 - val_lr: 4.0000e-05
# 
# Training complete in 54m 29s
# =============================================================================






#%% STEP 5: Evaluation with Annoy Indexer

# Find the accuracies of classes
classes_acc = dict() 

for i in range(len(y_test_int)):
    classes_acc[id2class_custom_valids[y_test_int[i]]] = 0

# Find Accuracies 
top1_accuracy, top10_accuracy, top20_accuracy, top50_accuracy = 0, 0, 0, 0

# Predictions
predictions = LSTM_model.predict(X_test, batch_size=512, verbose=1)
y_pred = []

for i in tqdm(range(len(predictions))):

    original_class = y_test_int[i]
    # Find the approximate neighbors with Annoy
    approximate_neighbors = word2vec_model.most_similar([predictions[i]], topn=300, indexer=annoy_index)
    close_words = []
    for neighbor in approximate_neighbors:
        close_words.append(neighbor[0])

    # Find the classes that are involved in approximate neighbors
    ranked_classes = []
    for potential_class in close_words:
        if potential_class in id2class_custom_valids.values():
            ranked_classes.append(potential_class)
    
    y_pred.append(ranked_classes[0])
    
    # Top-1 Accuracy
    if id2class_custom_valids[original_class] == ranked_classes[0]:
        top1_accuracy += 1
        classes_acc[id2class_custom_valids[original_class]] += 1
        
    # Top-10 Accuracy
    if id2class_custom_valids[original_class] in ranked_classes[:10]:
        top10_accuracy += 1
                
    # Top-20 Accuracy
    if id2class_custom_valids[original_class] in ranked_classes[:20]:
        top20_accuracy += 1
 
    # Top-50 Accuracy
    if id2class_custom_valids[original_class] in ranked_classes[:50]:
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
    classes_distribution_test[id2class_custom_valids[y_test_int[i]]] = 0

for i in range(len(y_test_int)):
    classes_distribution_test[id2class_custom_valids[y_test_int[i]]] += 1


counter = 0

for key in classes_acc:
    classes_acc[key] /= classes_distribution_test[key]
    if classes_acc[key] != 0:
        counter += 1
        
print('Number of classes with at least one correct prediction:', counter) #201


# Confusion Matrix
y_true = []

for i in range(len(y_test_int)):
    y_true.append(id2class_custom_valids[y_test_int[i]])
    

classes_names = []
for key in id2class_custom_valids:
    if id2class_custom_valids[key] not in classes_names:
        classes_names.append(id2class_custom_valids[key])


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

LSTM_model.save(os.path.join(save_output_dir, 'LSTM_model_30.6.h5'))

model = load_model(os.path.join(save_output_dir, 'LSTM_model_30.6.h5'))




#%% Plot Learning curves
with open(os.path.join(save_output_dir, 'history.pkl'), 'rb') as fp:
    train_history = pickle.load(fp)

loss_list        = train_history['loss']
cos_sim_list     = train_history['cosine_similarity']
lr_list          = train_history['lr']
val_loss_list    = train_history['val_loss']
val_cos_sim_list = train_history['val_cosine_similarity']
val_lr_list      = train_history['val_lr']



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





