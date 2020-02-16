# Imports
# Set seeds for python, numpy and tensorflow for reproducibility
from numpy.random import seed
from tensorflow import set_random_seed
import os
os.environ['PYTHONHASHSEED'] = '0'
import random as rn
import tensorflow as tf
from keras import backend as K
# Force tensorflow to use a single thread (recommended for reproducibility)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
import keras
from keras.constraints import max_norm
from keras.layers import Conv2D, MaxPooling2D, Reshape, Conv1D, MaxPooling1D, Concatenate, RNN
import gensim
from time import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Input, Embedding, Dense, Dropout, Bidirectional, TimeDistributed, Flatten
from keras import optimizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score, f1_score
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping


# Removes punctuation from input text
def clean_text(text):
    exclude = set(['.', ',', '?', '!'])
    text = ''.join(ch for ch in text if ch not in exclude)
    return text


# Train CNN model and evaluate performance.
def main():
    # Read in training data
    df = pd.read_csv('survey_data_1211.csv')
    
    # Remove testing review id = 6
    df = df[df['Participantid']!=6]
    
    # Bring review text

    text = pd.read_csv('review_pool.csv')
    
    df = df.merge(text, on = ['Id (Review)'], how = 'inner')
    df['ReviewText'] = df['ReviewText'].fillna('na')

    
    # Read in testing data
    #df_test = pd.read_csv('test_data.csv')
    #df_test['ReviewText'] = df_test['ReviewText'].fillna('na')

    

    # Clean test review text
    #df_test['ReviewText'] = df_test['ReviewText'].apply(clean_text)
    #df_test['ReviewText'] = df_test['ReviewText'].str.lower()

    # Clean training review text
    df['ReviewText'] = df['ReviewText'].apply(clean_text)
    df['ReviewText'] = df['ReviewText'].str.lower()
    
    
    # Create embedding matrix
    # Size of the dimensionality of the pre-trained word embeddings
    embedding_size = 300
    print('Loading pre-trained word embeddings..')
    t0 = time()
    file_name = 'GoogleNews-vectors-negative300.bin.gz'
    w2v = gensim.models.KeyedVectors.load_word2vec_format(file_name, binary=True)
    duration = time() - t0
    print("done in %fs" % (duration))
    print('creating embedding matrix..')

    

    # Map class names to numbers
    binary_rating_mapping = {'na': 0.0,
                             'True': 1.0}
    
    main_topic_names = ['Functionality', 'Rangeanxiety', 
                       'Availability', 'Cost', 'Userinteraction', 
                       'Location', 'Servicetime','Dealership']
    '''
    func_list = []
    range_list = []
    avail_list = []
    cost_list = []
    user_list = []
    location_list = []
    service_list = []
    dealer_list = []
    other_list = []
    '''
    
    #main_topic_names = ["Sentiment"]
    
    for i in list(range(0,len(main_topic_names))):    
        topic = main_topic_names[i]
    
        
        
        df[topic] = df[topic].fillna('na')
        df[topic] = df[topic].apply(str)
        df[topic] = df[topic].map(binary_rating_mapping)
    
        # Split data into train and test set
        # Split data into train and test set
    
    F1_list =[]
    ACC_list=[]
    
    num_iteration = 5
    for i in range(num_iteration):
            
        
        reviews_train, reviews_val, \
        y_train, y_val = train_test_split(df['ReviewText'].values,
                                           df[main_topic_names].values,
                                           test_size=0.1)
        
        reviews_val, reviews_test,\
        y_val, y_test = train_test_split(reviews_val, 
                                           y_val,
                                           test_size = 0.5)
    
    
        print('Train size: %s' % len(reviews_train))
    
        # Convert numpy arrays to lists
        reviews_train = np.array(reviews_train)
        reviews_val = np.array(reviews_val)
        y_train = np.array(y_train)
        y_val = np.array(y_val)
        
        
        # Map class names to numbers
        #binary_rating_mapping2 = {'na': 0.0,
        #                         '1.0': 1.0}
        
        '''
        for i in list(range(0,len(main_topic_names))):    
            topic = main_topic_names[i]
        
            df_test[topic] = df_test[topic].fillna('na')
            df_test[topic] = df_test[topic].apply(str)
            df_test[topic] = df_test[topic].map(binary_rating_mapping2)
        '''
    
        # Split data into train and test set
        # Split data into train and test set
    
        
        reviews_test = np.array(reviews_test)
        y_test = np.array(y_test)
    
    
        # Tell the tokenizer to use the entire vocabulary
        num_words = None
        tokenizer = Tokenizer(num_words=num_words, oov_token='Out_Of_Vocab_Token')
        tokenizer.fit_on_texts(reviews_train)
    
        # Now set number of words to the size of the vocabulary
        num_words = len(tokenizer.word_index)
    
        # Convert reviews to lists of tokens
        x_train_tokens = tokenizer.texts_to_sequences(reviews_train)
        x_val_tokens = tokenizer.texts_to_sequences(reviews_val)
        x_test_tokens = tokenizer.texts_to_sequences(reviews_test)
        
        
        # Pad all sequences of tokens to be the same length (length of the longest sequence)
        num_tokens = [len(tokens) for tokens in x_train_tokens]
        num_tokens = np.array(num_tokens)
        max_tokens = np.max(num_tokens)
    
        # Pad zeroes to the beginning of the sequences
        pad = 'pre'
        x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens, padding=pad, truncating=pad)
        x_val_pad = pad_sequences(x_val_tokens, maxlen=max_tokens, padding=pad, truncating=pad)
        x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens, padding=pad, truncating=pad)
       
    
        
    
        # Good explaination of this at https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
        num_missing = 0
        # indices of rows in embedding matrix that aren't initialized (because the corresponding word was not in word2vec)
        missing_word_indices = []
        embedding_matrix = np.zeros((num_words + 1, embedding_size))
        for word, i in tokenizer.word_index.items():
            if word in w2v.vocab:
                embedding_vector = w2v[word]
                embedding_matrix[i] = embedding_vector
            else:
                num_missing += 1
                missing_word_indices.append(i)
    
        # Fill in uninitialized rows of embedding matrix with random numbers. 0.25 is chosen so these vectors
        # have approximately the same variance as the pre-trained word2vec ones
        random_vectors = np.random.uniform(-0.25, 0.25, (num_missing, embedding_size))
        for i in range(num_missing):
            embedding_matrix[missing_word_indices[i]] = random_vectors[i]
    
        num_cell = 64
        dropout_percent = 0.6
        rdropout = 0.2
        batch_size = 32
        num_epoch = 20
    
        # Build model    
        model = Sequential()
        model.add(Embedding(input_dim=num_words + 1, output_dim=embedding_size,
                               weights=[embedding_matrix], trainable=True,
                               input_length=max_tokens, name='embedding_layer'))
        model.add(LSTM(num_cell, dropout=dropout_percent, recurrent_dropout=rdropout))
        model.add(Dense(8, activation='sigmoid'))
        
        # try using different optimizers and different optimizer configs
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['categorical_accuracy'])
        
        print('Train...')
        t0 = time()
        filepath="rnn_multilabel.hdf5"
    
        early_stop = EarlyStopping(monitor='loss', patience=5, verbose=0, mode='min')
        checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=0, save_best_only=True, mode='max')
    
        history = model.fit(x_train_pad, y_train,
                  batch_size=batch_size,
                  epochs=num_epoch,
                  validation_data=(x_val_pad, y_val),
                  callbacks=[early_stop, checkpoint])
        duration = time() - t0
    
        score, acc = model.evaluate(x_val_pad, y_val,
                                    batch_size=batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)
        
    
    
        # Train model
    
    
        model.save(filepath)
    
    
    
        # Generate predictions on test set
        
        t0 = time()
        
        print("done in %fs" % (duration))
        print('Generating predictions on the test set...\n')
        y_pred = model.predict(x_test_pad)
        y_pred_class = np.round(y_pred, 0)
        
        prediction_time = time() - t0
    
        print('Batch Size: %i' % batch_size)
        print('Num Epoch: %i' % num_epoch)
        print('No. Cell: %i' %num_cell)
        print('Dropout: %.2f' %dropout_percent)
        print('Recurrent Dropout: %.2f' %rdropout)    
        
        F1 = f1_score(y_true=y_test, y_pred=y_pred_class, average='weighted')
        ACC = accuracy_score(y_test, y_pred_class)
        
        F1_list.append(F1)
        ACC_list.append(ACC)
        
        for i in list(range(0,len(main_topic_names))):    
            topic = main_topic_names[i]
        
        
            # Evaluate model performance
            print(topic)
            accuracy = 100 * accuracy_score(y_test[:,i], y_pred_class[:,i])
            precision = precision_score(y_test[:,i], y_pred_class[:,i])
            recall = recall_score(y_test[:,i], y_pred_class[:,i])
            f1 = 2*precision*recall/(precision+recall)
            print('Accuracy: %.2f%%' % (accuracy))
            print('Precision: %.2f' % precision)
            print('Recall: %.2f' % recall )
            print('F1 Score: %.2f' % f1)
            print('Train Time: %f' % duration)
            print('Prediction Time: %f' % prediction_time)
            
            result = pd.DataFrame({'Category': [topic],'Accuracy':[accuracy],'Precision':[precision], 'Recall': [recall], 'F1 Score': [f1], 'Training Time': [duration], 'Prediction Time': [prediction_time]})
        
            result.to_csv('rnn_multi_uncertainty.csv', index=True, mode=  'a', header=False)
            
        ACC_list
        F1_list
        print("ACC_min: %.2f" % min(ACC_list))
        print("ACC_ave: %.2f" % (sum(ACC_list)/len(ACC_list)))
        print("ACC_max: %.2f" % max(ACC_list))
        
        print("F1_min: %.2f" % min(F1_list))
        print("F1_ave: %.2f" % (sum(F1_list)/len(F1_list)))
        print("F1_max: %.2f" % max(F1_list))    

    



if __name__ == '__main__':
    main()
