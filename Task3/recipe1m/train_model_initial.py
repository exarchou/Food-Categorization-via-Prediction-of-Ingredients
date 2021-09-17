# =============================================================================
# Script to train and evaluate an LSTM model for the classification problem of Recipe1M
# 5. Simple version with bi LSTM, trainable EM and cos. sim. as cost function
# Author: Dimitrios-Marios Exarchou
# Last modified: 13/9/2021   
# 
# Top-1   Accuracy = 28.70%
# Top-10  Accuracy = 43.58%
# Top-20  Accuracy = 48.32%
# Top-50  Accuracy = 55.87%
#
# Training complete in 7m 32s
# =============================================================================




#%% STEP 1: Loading Libraries
import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras import metrics
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.initializers import Constant
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger,ReduceLROnPlateau
from keras.optimizers import Adam
from gensim.models import KeyedVectors
from gensim.similarities.annoy import AnnoyIndexer
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight




#%% STEP 2: Loading necessary files
# Directories

save_output_dir = 'X:/thesis_experiments/initial/5'

EMBEDDING_VECTOR_LENGTH = 300
MAX_SEQUENCE_LENGTH = 20

ingr_vocab = pickle.load(open('X:/thesis_experiments/initial/data_initial/ingr_vocab.pkl', 'rb'))
embedding_matrix = pickle.load(open('X:/thesis_experiments/initial/data_initial/word2vec_data/embedding_matrix.pkl', 'rb'))
id2class_custom  = pickle.load(open('X:/thesis_experiments/initial/data_initial/id2class_reloaded.pkl', 'rb'))

X_train = pickle.load(open('X:/thesis_experiments/initial/data_initial/X_train_word2vec.pkl', 'rb'))
y_train = pickle.load(open('X:/thesis_experiments/initial/data_initial/y_train_word2vec.pkl', 'rb'))
X_valid = pickle.load(open('X:/thesis_experiments/initial/data_initial/X_valid_word2vec.pkl', 'rb'))
y_valid = pickle.load(open('X:/thesis_experiments/initial/data_initial/y_valid_word2vec.pkl', 'rb'))
X_test  = pickle.load(open('X:/thesis_experiments/initial/data_initial/X_test_word2vec.pkl', 'rb'))   
y_test  = pickle.load(open('X:/thesis_experiments/initial/data_initial/y_test_word2vec.pkl', 'rb'))
y_test_int = np.loadtxt('X:/thesis_experiments/initial/data_initial/y_test_int.txt', dtype=int)


word2vec_model = KeyedVectors.load_word2vec_format(('X:/thesis_experiments/initial/data_initial/word2vec_data/GoogleNews-vectors-negative300.bin'), binary=True)
annoy_index = AnnoyIndexer()
annoy_index.load('X:/thesis_experiments/initial/data_initial/word2vec_data/annoy_index')




#%% STEP 3: Building LSTM model
LSTM_model = Sequential()
LSTM_model.add(Embedding(len(ingr_vocab),                                   # number of unique tokens
                         EMBEDDING_VECTOR_LENGTH,                           # length of embedded vectors
                         embeddings_initializer=Constant(embedding_matrix), # initialization of matrix 
                         input_length=MAX_SEQUENCE_LENGTH,                  # max number of ingredients
                         trainable=True))
LSTM_model.add(Bidirectional(LSTM(128, dropout=0.0, recurrent_dropout=0.0, return_sequences=True)))
LSTM_model.add(Bidirectional(LSTM(128, dropout=0.0, recurrent_dropout=0.0)))
LSTM_model.add(Dense(EMBEDDING_VECTOR_LENGTH, activation='tanh'))

LSTM_model.summary()




#%% STEP 4: Training Process
since = time.time()  

# Optimizer
optimizer = Adam(learning_rate=0.002)

# Model Compile
LSTM_model.compile(loss='cosine_similarity', optimizer=optimizer, metrics=[metrics.MeanSquaredError()])

# Callbacks
filepath = os.path.join(save_output_dir, 'weights-{epoch:02d}-{abs(val_loss):.4f}.hdf5')                                               # File name includes epoch and validation accuracy
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, mode='min', verbose=1)                                 # Use Mode = max for accuracy and min for loss
early_stop = EarlyStopping(monitor='val_loss', mode='min', min_delta = 0.00025, patience=8, restore_best_weights=True, verbose=1)      # Early Stopping if val loss does not drop for at least 0.00025 in 10 epochs.                
log_csv = CSVLogger(os.path.join(save_output_dir, 'my_logs2csv') , separator='\t', append=True)                                        # CSVLogger logs epoch, acc, loss, val_acc, val_loss
reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.2, patience=4, min_delta=0.00025, verbose=1)                    # Redule lr if val loss does not drop for at least 0.00025 at 4 epochs.
callbacks_list = [checkpoint, early_stop, log_csv, reduce_lr]


# Model Train
history = LSTM_model.fit(X_train, y_train, epochs=24, batch_size=2048, validation_data=(X_valid, y_valid), verbose=1, callbacks=callbacks_list) 

time_elapsed = time.time() - since
print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))   
 



#%% STEP 5: Evaluation with Annoy Indexer
# Initialize a zero vector for the accuracy of the classes
classes_acc = dict() 

for i in range(len(y_test_int)):
    classes_acc[id2class_custom[y_test_int[i]]] = 0

# Find top-k Accuracies 
top1_accuracy, top10_accuracy, top20_accuracy, top50_accuracy = 0, 0, 0, 0

# Predictions
predictions = LSTM_model.predict(X_test, batch_size=512, verbose=1)
y_pred = []

for i in tqdm(range(len(predictions))):

    # Read the original class
    original_class = y_test_int[i]
    
    # Find the 300 approximate neighbors with Annoy
    approximate_neighbors = word2vec_model.most_similar([predictions[i]], topn=300, indexer=annoy_index)
    close_words = []
    for neighbor in approximate_neighbors:
        close_words.append(neighbor[0])

    # Find the classes that are involved in approximate neighbors
    ranked_classes = []
    for potential_class in close_words:
        if potential_class in id2class_custom.values():
            ranked_classes.append(potential_class)
    
    y_pred.append(ranked_classes[0])
    
    # Top-1 Accuracy
    if id2class_custom[original_class] == ranked_classes[0]:
        top1_accuracy += 1
        classes_acc[id2class_custom[original_class]] += 1
        
    # Top-10 Accuracy
    if id2class_custom[original_class] in ranked_classes[:10]:
        top10_accuracy += 1
                
    # Top-20 Accuracy
    if id2class_custom[original_class] in ranked_classes[:20]:
        top20_accuracy += 1
 
    # Top-50 Accuracy
    if id2class_custom[original_class] in ranked_classes[:50]:
        top50_accuracy += 1

            
print('\n' + '=' * 30)
print(f'Top-1   Accuracy = {  top1_accuracy * 100 / len(X_test) :.2f}%')
print(f'Top-10  Accuracy = { top10_accuracy * 100 / len(X_test) :.2f}%')
print(f'Top-20  Accuracy = { top20_accuracy * 100 / len(X_test) :.2f}%')
print(f'Top-50  Accuracy = { top50_accuracy * 100 / len(X_test) :.2f}%')
print('=' * 30)




#%% STEP 6: Accuracy of Classes & Confusion Matrix
classes_distribution_test = dict()

for i in range(len(y_test_int)):
    classes_distribution_test[id2class_custom[y_test_int[i]]] = 0

for i in range(len(y_test_int)):
    classes_distribution_test[id2class_custom[y_test_int[i]]] += 1

counter = 0

for key in classes_acc:
    classes_acc[key] /= classes_distribution_test[key]
    if classes_acc[key] != 0:
        counter += 1
        
print('Number of classes with at least one correct prediction:', counter) # 250

# Confusion Matrix
y_true = []

for i in range(len(y_test_int)):
    y_true.append(id2class_custom[y_test_int[i]])
    
classes_names = []

for key in id2class_custom:
    if id2class_custom[key] not in classes_names:
        classes_names.append(id2class_custom[key])

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
mse_list         = train_history['mean_squared_error']
cos_sim_list     = train_history['loss']
val_mse_list     = train_history['val_mean_squared_error']
val_cos_sim_list = train_history['val_loss']

# Plot Learning curves
plt.plot(cos_sim_list, label='Train. Cos.Sim.')
plt.plot(val_cos_sim_list, label='Val. Cos.Sim.')
plt.legend(frameon=False)
plt.savefig(os.path.join(save_output_dir, 'Cos.Sim.jpg'), transparent=True, dpi=200)
plt.show()

plt.plot(mse_list, label='Train. MSE')
plt.plot(val_mse_list, label='Val. MSE')
plt.legend(frameon=False)
plt.savefig(os.path.join(save_output_dir, 'MSE.jpg'), transparent=True, dpi=200)
plt.show()




#%% STEP 8: Print Recall, Precision and F1-score
# Unique Classes' Names
classes_names = []

for key in id2class_custom:
    if id2class_custom[key] not in classes_names:
        classes_names.append(id2class_custom[key])

# Calculate Recall and Precision
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

#%% Find the number of instances of zero-accuracy classes
counter_of_zero_accuracy_classes = 0

for key in classes_acc:
    if classes_acc[key] == 0:
        counter_of_zero_accuracy_classes += classes_distribution_test[key]

print('Number of instances of zero-accuracy classes:', counter_of_zero_accuracy_classes) # 13210




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




# #%% STEP 10: Load saved stats, confusion matrix and model
# model = load_model(filepath)

# with open(os.path.join(save_output_dir, 'classes_acc.pkl'), 'rb') as fp:   
#     classes_acc = pickle.load(fp)

# with open(os.path.join(save_output_dir, 'classes_distribution_test.pkl'), 'rb') as fp:    
#     classes_distribution_test = pickle.load(fp)

# with open(os.path.join(save_output_dir, 'confusion_matrix.pkl'), 'rb') as fp:    
#     confusion_matrix = pickle.load(fp)

# with open(os.path.join(save_output_dir, 'confusion_matrix_dataframe.pkl'), 'rb') as fp:     
#     confusion_matrix_dataframe = pickle.load(fp)

# with open(os.path.join(save_output_dir, 'history.pkl'), 'rb') as fp:
#     train_history = pickle.load(fp)

# loss_list        = train_history['loss']
# cos_sim_list     = train_history['cosine_similarity']
# val_loss_list    = train_history['val_loss']
# val_cos_sim_list = train_history['val_cosine_similarity']


