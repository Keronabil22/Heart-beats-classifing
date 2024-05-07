#!/usr/bin/env python
# coding: utf-8

#  libraries

# In[1]:


import wfdb #waveform database 
# from wfdb import processing
import os
import pandas as pd
import glob as gb
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import tensorflow as tf
# from scipy import signal
from sklearn.model_selection import train_test_split #to split data
import pywt #pywavelet lib
# from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Dropout , LSTM , Bidirectional
from keras.optimizers import Adam
from tensorflow.math import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns

# functions

#     1-for Collecting data

# In[2]:


#to find paths for all records using my database , the files should have attributes
def find_paths (file_paths):
    file_paths = gb.glob(file_paths + '/*.atr')
    file_paths = [os.path.splitext(path)[0].replace("\\", "/") for path in file_paths]
    return file_paths


# In[3]:


#for spliting all records into beats
def split_records(record_list):
    beats=[]
    annotation_symbols=[]
    for record_Name in record_list :
        record=wfdb.rdrecord(record_Name)
        ecg_signal,_=wfdb.rdsamp(record_Name)
        ecg_signal=filter_ecg (ecg_signal)
        ecg_signal=normalize_ecg (ecg_signal)
        annotation = wfdb.rdann(record_Name, 'atr', sampfrom=0, sampto=record.sig_len)
        # Get the indices of the QRS complexes (peaks)
        qrs_inds = annotation.sample
        # Define the duration (in samples) for each beat segment
        beat_duration = int(0.2 * record.fs)  # Assuming a 200 ms duration for each beat
        for qrs_ind in qrs_inds:
            start = max(0, qrs_ind - beat_duration // 2)  # Start of the beat segment
            end = min(qrs_ind + beat_duration // 2, record.sig_len)  # End of the beat segment
            beat = record.p_signal[start:end,0]  # Extract the beat segment
            beat=beat.tolist()
            if len(beat)<50:
                a=beat[-1]
                for i in range(50-len(beat)):
                    beat.append(a)
            beat=np.array(beat)
            beats.append(beat)
            beat=[]
        annotation_symbols.append((annotation.symbol))
    beats=np.array(beats)
    annotation_symbols = [item for sublist in annotation_symbols for item in sublist]
    annotation_symbols,annotation_symbols_dic,symb_num = encoder(annotation_symbols)
    annotation_symbols=np.array(annotation_symbols)
    data=make_df(beats,annotation_symbols)
    
    return beats , annotation_symbols,annotation_symbols_dic,symb_num


# In[4]:


def make_df (beats , Labels):
    beat_data=pd.Series(beats.tolist(),name='beats')
    ann=pd.Series(Labels,name='annotations')
    data=pd.concat([beat_data,ann],axis=1)
    data=data.sample(frac=1).reset_index(drop=True)
    return data


# 2-Translate Labels

# In[5]:


#encoding the labels in training stage (the fitting statment code doesn't accept strings as a labels)
def encoder (labels):
    my_set_labels=set(labels)
    my_set_numbers=set(range(len(my_set_labels)))
    my_dic = dict(zip(my_set_labels,my_set_numbers))
    converted_list = [my_dic.get(item, item) for item in labels]
    return converted_list , my_dic , len(set(converted_list))


# In[6]:


#decoding the labels in testing stage
def decoder (labels,dic):
    labels=[next(key for key, value in dic.items() if value == t ) for t in labels]
    return labels


# In[7]:


#for removing noise from data
def filter_ecg (ecg_signal):
    # Apply wavelet transform
    wavelet = 'db6'  # Choose the desired wavelet
    coeffs = pywt.wavedec(ecg_signal, wavelet) #decomposing signal 

    # Set a threshold for noise removal
    threshold = 0.5  # Adjust according to your signal characteristics

    # Apply thresholding to remove noise
    denoised_coeffs = [pywt.threshold(c, threshold) for c in coeffs]

    # Reconstruct the denoised signal
    denoised_signals = pywt.waverec(denoised_coeffs, wavelet)
    return denoised_signals


# In[8]:


#for normalizing data
def normalize_ecg (ecg_signal):
    normalize_signal=(ecg_signal-np.mean(ecg_signal))/np.std(ecg_signal)
    return normalize_signal


# 3- for Abnormal Data

# In[9]:


def filter_data (data):
    # Assuming 'data' is your DataFrame
    # Find the majority value
    majority_value = data['annotations'].value_counts().idxmax()

    # Filter the DataFrame to exclude rows with the majority value
    abnormal_data = data[data['annotations'] != majority_value]
    normal_data=data[data['annotations'] == majority_value]
    
    return abnormal_data , normal_data


# In[10]:


def increase_data (abnormal_data,normal_data):
    # Data augmentation for minority class
    augmented_data1 = []
    augmented_data2 = []
    for index, row in abnormal_data.iterrows():
        signal = row['beats']  # Assuming 'Data' column contains the signal
        # Apply augmentation techniques
        augmented_signal1 = frequency_transform(signal, noise_level=0.01)
#         augmented_signal2 = envelope_interpolation(signal, interpolation_factor=2)
        augmented_data1.append(augmented_signal1)
#         augmented_data2.append(augmented_signal2)
    # Create DataFrame from augmented data
    augmented_df1 = pd.DataFrame({'beats': augmented_data1, 'annotations': abnormal_data['annotations']})
#     augmented_df2 = pd.DataFrame({'beats': augmented_data2, 'annotations': abnormal_data['annotations']})
    abnor_data= pd.concat([abnormal_data, augmented_df1])
    abnor_data= abnor_data.sample(frac=1.0)
    normal_data = normal_data.sample(n=len(abnor_data))
    final_data=pd.concat([abnor_data,normal_data])
    final_data=final_data.sample(frac=1.0,random_state=42)
    return final_data


# In[11]:


# Frequency Domain Transformations (Applying Gaussian Noise)
def frequency_transform(signal, noise_level=0.01):
    noise = np.random.normal(0, noise_level, len(signal))
    fft_signal = np.fft.fft(signal)
    fft_signal += noise
    return np.fft.ifft(fft_signal).real


# 4-Splitting data

# In[12]:


#for spliting thim into train,test&valid
def spliting_data (data):
    beats = np.array(data.iloc[:,0])
    annotation_symbols=np.array(data.iloc[:,1])
    beats = np.array([np.array(sublist) for sublist in beats])
    X_train, X_test, Y_train, Y_test = train_test_split(beats,annotation_symbols, test_size=0.3, random_state=42)
    X_test, X_valid, Y_test, Y_valid = train_test_split(X_test,Y_test, test_size=0.5, random_state=42)
    return X_train ,X_test , X_valid , Y_train , Y_test , Y_valid


# 5-Models

# In[13]:


def RNN_model (X_train,Y_train , X_valid , Y_valid , no_symbols,epochs):
     
    RNN_model = Sequential([
            SimpleRNN(units=128, input_shape=(50, 1), activation='relu', return_sequences=True),  
            Dropout(0.2),
            SimpleRNN(units=128, activation='relu', return_sequences=True),
            Dropout(0.2),
            SimpleRNN(units=128, activation='relu'),  
            Dropout(0.2),
            Dense(no_symbols, activation='softmax')  # Assuming no_symbols is the number of classes
        ])


    # Compile the model
    RNN_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=1e-3),
                  metrics=['accuracy'])
    #ModelSummary
    RNN_summary=RNN_model.summary()
    # Train the model
    RNN_hist_model = RNN_model.fit(X_train, Y_train,
                                  epochs=epochs , validation_data=(X_valid, Y_valid))
    return RNN_summary,RNN_model,RNN_hist_model


# In[14]:


def LSTM_model (X_train,Y_train , X_valid , Y_valid , no_symbols,epochs):

    LSTM_model = Sequential([
    (Bidirectional(LSTM(units=128,return_sequences=True),input_shape=(50, 1))),  # LSTM layer with 64 units
    (Dropout(0.2)),
    (Bidirectional(LSTM(units=64,return_sequences=True))),  # LSTM layer with 64 units
    Dropout(0.2),
    Bidirectional(LSTM(units=32)),  # Third LSTM layer with 64 units
    Dense(units=no_symbols , activation='softmax')   # Output layer with 5 units (one for each class)
    ])

    # Compile the model
    LSTM_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    LSTM_summary=LSTM_model.summary()
    # Train the model
    LSTM_hist_model = LSTM_model.fit(X_train, Y_train,
                                  epochs=epochs, batch_size=32,
                                  validation_data=(X_valid, Y_valid))
    return LSTM_summary, LSTM_model,LSTM_hist_model


# 6-Evaluation

# In[15]:


def evaluate_model(model,X_test,Y_test):
    test_loss, test_accuracy = model.evaluate(X_test, Y_test)
    print("Test Accuracy:", test_accuracy*100)
    print("Test Loss:", test_loss)


# In[16]:


def predict (X_test,model,dic):
    y_pred=model.predict(X_test)
    y_pred=np.argmax(y_pred,axis=1)
    y_pred=decoder(y_pred,dic)
    return y_pred


# 7-Statistical Graphs

# In[17]:


def Loss_comp_graph (h_Model1,h_Model2,model1_name,model2_name):
    plt.figure(figsize=(12, 5))
    plt.plot(h_Model1.history['loss'], label=f'{model1_name} Train')
    plt.plot(h_Model1.history['val_loss'], label=f'{model1_name} Validation')
    plt.plot(h_Model2.history['loss'], label=f'{model2_name} Train')
    plt.plot(h_Model2.history['val_loss'], label=f'{model2_name} Validation')
    plt.title('Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# In[18]:


def Accuracy_comp_graph (h_Model1,h_Model2,model1_name,model2_name):
    plt.figure(figsize=(12, 5))
    plt.plot(h_Model1.history['accuracy'], label=f'{model1_name} Train')
    plt.plot(h_Model1.history['val_accuracy'], label=f'{model1_name} Validation')
    plt.plot(h_Model2.history['accuracy'], label=f'{model2_name} Train')
    plt.plot(h_Model2.history['val_accuracy'], label=f'{model2_name} Validation')
    plt.title('Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    
def conf_mat (Y_test , Y_pred,dic):
    Y_pred_encoded=[]
    for i in Y_pred :
        Y_pred_encoded.append(dic[i])
    conf_matrix = confusion_matrix(Y_test , Y_pred_encoded)
    plt.figure(figsize=(15,7))
    sns.heatmap(conf_matrix,annot=True , fmt='d',cmap='Blues')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    return Y_pred_encoded

def classification_report (Y_test , Y_pred):
    print(classification_report(Y_test,Y_pred))

    

