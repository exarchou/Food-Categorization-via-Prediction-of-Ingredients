# =============================================================================
# Script to train and evaluate an LSTM model for the classification problem of Food-101
# Author: Dimitrios-Marios Exarchou
# Last modified: 13/9/2021   
# =============================================================================


# Arguments
dropout           = 0.30
recurrent_dropout = 0.30
units             = 256
batch_size        = 512





#%% STEP 1: Loading Libraries
import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from keras.initializers import Constant
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.optimizers import Adam
from gensim.models import KeyedVectors
from gensim.similarities.annoy import AnnoyIndexer
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight





#%% STEP 2: Loading necessary files
# Directories
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

# Create a file
file_object = open(os.path.join(save_output_dir, 'meta.txt'), 'a')





#%% STEP 3: Building LSTM model
LSTM_model = Sequential()
LSTM_model.add(Embedding(len(ingr_vocab),                                   # number of unique tokens
                         EMBEDDING_VECTOR_LENGTH,                           # length of embedded vectors
                         embeddings_initializer=Constant(embedding_matrix), # initialization of matrix 
                         input_length=MAX_SEQUENCE_LENGTH,                  # max number of ingredients
                         trainable=True))
LSTM_model.add(Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True)))
LSTM_model.add(Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout)))
LSTM_model.add(Dense(EMBEDDING_VECTOR_LENGTH, activation='tanh'))

LSTM_model.summary()





#%% STEP 4: Training Process
since = time.time()  

# Optimizer
optimizer = Adam(learning_rate=0.002)

# Model Compile
LSTM_model.compile(loss='cosine_similarity', optimizer=optimizer)

# Callbacks
filepath = os.path.join(save_output_dir, 'weights-{epoch:02d}-{val_loss:.4f}.hdf5')                                                    # File name includes epoch and validation accuracy
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, mode='min', verbose=1)                                 # Use Mode = max for accuracy and min for loss
early_stop = EarlyStopping(monitor='val_loss', mode='min', min_delta = 0.00025, patience=6, restore_best_weights=True, verbose=1)      # Early Stopping if val loss does not drop for at least 0.00025 in 6 epochs.                
log_csv = CSVLogger(os.path.join(save_output_dir, 'my_logs2csv') , separator='\t', append=True)                                        # CSVLogger logs epoch, acc, loss, val_acc, val_loss
reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.2, patience=3, min_delta=0.00025, verbose=1)                    # Redule lr if val loss does not drop for at least 0.00025 at 3 epochs.
callbacks_list = [checkpoint, early_stop, log_csv, reduce_lr]


# Model Train
history = LSTM_model.fit(X_train, y_train, epochs=24, batch_size=batch_size, validation_data=(X_valid, y_valid), verbose=1, callbacks=callbacks_list) 

time_elapsed = time.time() - since
print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))   
file_object.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))





#%% STEP 5: Evaluation with Annoy Indexer
# Initialize a zero vector for the accuracy of the classes
classes_acc = dict() 

for i in range(len(y_test_int)):
    classes_acc[classes_reloaded[y_test_int[i]]] = 0

# Find Accuracies 
top1_accuracy, top5_accuracy, top10_accuracy, top20_accuracy = 0, 0, 0, 0

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
    if classes_reloaded[original_class] in ranked_classes[:5]:
        top5_accuracy += 1
                
    # Top-20 Accuracy
    if classes_reloaded[original_class] in ranked_classes[:10]:
        top10_accuracy += 1
 
    # Top-50 Accuracy
    if classes_reloaded[original_class] in ranked_classes[:20]:
        top20_accuracy += 1

            
print('\n' + '=' * 30)
print(f'Top-1   Accuracy = {  top1_accuracy * 100 / len(X_test) :.2f}%')
print(f'Top-5  Accuracy = { top10_accuracy * 100 / len(X_test) :.2f}%')
print(f'Top-10  Accuracy = { top20_accuracy * 100 / len(X_test) :.2f}%')
print(f'Top-20  Accuracy = { top50_accuracy * 100 / len(X_test) :.2f}%')
print('=' * 30)
file_object.write(f'Top-1   Accuracy = {  top1_accuracy * 100 / len(X_test) :.2f}%\n')
file_object.write(f'Top-5  Accuracy = { top10_accuracy * 100 / len(X_test) :.2f}%\n')
file_object.write(f'Top-10  Accuracy = { top20_accuracy * 100 / len(X_test) :.2f}%\n')
file_object.write(f'Top-20  Accuracy = { top50_accuracy * 100 / len(X_test) :.2f}%\n')





#%% STEP 6: Accuracy of Classes & Confusion Matrix
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
        
print('Number of classes with at least one correct prediction:', counter) 
file_object.write(f'Number of classes with at least one correct prediction = {counter :d}\n')

# Confusion Matrix
y_true = []

for i in range(len(y_test_int)):
    y_true.append(classes_reloaded[y_test_int[i]])
    

classes_names = []
for key in classes_reloaded:
    if key not in classes_names:
        classes_names.append(key)

confusion_matrix = confusion_matrix(y_true, y_pred, labels=classes_names)

# Measure top-1 Accuracy from the diagonal of confusion matrix for validation
accuracy = 0

for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        if i == j:
            accuracy += confusion_matrix[i][j]

accuracy /= np.sum(confusion_matrix)
accuracy *= 100
print(f'Top-1   Accuracy = {  accuracy:.2f} %')

# Transform confusion matrix into dataframe
cm_df = pd.DataFrame(confusion_matrix, columns=classes_names, index=classes_names)





#%% STEP 7: Plot Learning Curves
train_history = history.history
cos_sim_list     = train_history['loss']
val_cos_sim_list = train_history['val_loss']

# Plot Learning curves
plt.plot(cos_sim_list, label='Train. Cos.Sim.')
plt.plot(val_cos_sim_list, label='Val. Cos.Sim.')
plt.legend(frameon=False)
plt.savefig(os.path.join(save_output_dir, 'Cos.Sim.jpg'), transparent=True, dpi=200)
plt.show()





#%% STEP 8: Print Recall, Precision and F1-score
recall    = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis = 1)
precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis = 0)

# Handle NaN values (divisions with zero)
recall[np.isnan(recall)], precision[np.isnan(precision)] = 0, 0

# Find Average Recall and Precision
recall_avg, precision_avg = np.mean(recall), np.mean(precision)
f_score = 2 * precision_avg * recall_avg / (precision_avg + recall_avg)

print(f' Recall    = { recall_avg * 100    :.2f}%')
print(f' Precision = { precision_avg * 100 :.2f}%')
print(f' F-score   = { f_score * 100       :.2f}%')
file_object.write(f'Recall    = { recall_avg * 100    :.2f}%\n')
file_object.write(f'Precision = { precision_avg * 100 :.2f}%\n')
file_object.write(f'F-score   = { f_score * 100       :.2f}%\n')


# Find Average Recall and Precision only for the classes that have at least one correct prediction 
counter_non_zeros = 0

for i in range(len(recall)):  
    if recall[i] != 0:     
        counter_non_zeros += 1
        
recall_avg_nonzeros = np.sum(recall) / counter_non_zeros

counter_non_zeros = 0

for i in range(len(precision)): 
    if precision[i] != 0:     
        counter_non_zeros += 1

precision_avg_nonzeros = np.sum(precision) / counter_non_zeros
f_score_non_zeros = 2 * precision_avg_nonzeros * recall_avg_nonzeros / (precision_avg_nonzeros + recall_avg_nonzeros)

print(f' Recall of non-zeros    = { recall_avg_nonzeros * 100    :.2f}%')
print(f' Precision of non-zeros = { precision_avg_nonzeros * 100 :.2f}%')
print(f' F-score of non-zeros   = { f_score_non_zeros * 100      :.2f}%')
file_object.write(f'Recall of non-zeros    = { recall_avg_nonzeros * 100    :.2f}%\n')
file_object.write(f'Precision of non-zeros = { precision_avg_nonzeros * 100 :.2f}%\n')
file_object.write(f'F-score of non-zeros   = { f_score_non_zeros * 100      :.2f}%\n')


#%% Find the number of instances of zero-accuracy classes
counter_of_zero_accuracy_classes = 0

for key in classes_acc:
    if classes_acc[key] == 0:
        counter_of_zero_accuracy_classes += classes_distribution_test[key]

print('Number of instances of zero-accuracy classes:', counter_of_zero_accuracy_classes) 
file_object.write(f'Number of instances of zero-accuracy classes = {counter_of_zero_accuracy_classes :d}\n')
file_object.close()





#%% STEP 9: Save stats, confusion matrix and model
with open(os.path.join(save_output_dir, 'classes_acc.pkl'), 'wb') as fp:   
    pickle.dump(classes_acc, fp)

with open(os.path.join(save_output_dir, 'classes_distribution_test.pkl'), 'wb') as fp:    
    pickle.dump(classes_distribution_test, fp)

with open(os.path.join(save_output_dir, 'confusion_matrix.pkl'), 'wb') as fp:    
    pickle.dump(confusion_matrix, fp)

with open(os.path.join(save_output_dir, 'confusion_matrix_dataframe.pkl'), 'wb') as fp:     
    pickle.dump(cm_df, fp)

with open(os.path.join(save_output_dir, 'history.pkl'), 'wb') as fp:
    pickle.dump(history.history, fp)

filepath = os.path.join(save_output_dir, str('LSTM_model_{:.2f}.h5'.format(accuracy)))       
LSTM_model.save(filepath)





# =============================================================================
# #%% STEP 10: Load saved stats, confusion matrix and model
# model = load_model(filepath)
# 
# with open(os.path.join(save_output_dir, 'classes_acc.pkl'), 'rb') as fp:   
#     classes_acc = pickle.load(fp)
# 
# with open(os.path.join(save_output_dir, 'classes_distribution_test.pkl'), 'rb') as fp:    
#     classes_distribution_test = pickle.load(fp)
# 
# with open(os.path.join(save_output_dir, 'confusion_matrix.pkl'), 'rb') as fp:    
#     confusion_matrix = pickle.load(fp)
# 
# with open(os.path.join(save_output_dir, 'confusion_matrix_dataframe.pkl'), 'rb') as fp:     
#     confusion_matrix_dataframe = pickle.load(fp)
# 
# with open(os.path.join(save_output_dir, 'history.pkl'), 'rb') as fp:
#     train_history = pickle.load(fp)
# =============================================================================
