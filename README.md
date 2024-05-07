# Heart-beats-classifing
    This Python project focuses on classifying heartbeat signals using various deep learning models. It involves preprocessing the data, building and training models, and evaluating their performance.

# Table of Contents
* 1-Introduction
* 2- DataBase
* 3- Project Structure
* 4- Pre-requirements 
* 5- Data Preprocessing
* 6- Model Building
* 7- Model Evaluation
* 8- Results and Analysis
* 9- Code Guide

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Introduction
    The classification of heart beat signals is a crucial task in healthcare for diagnosing various heart conditions. This project employs machine learning techniques to automatically classify heart beat signals into different categories such as normal beats and abnormal beats.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## DataBase
This database consists of 90 annotated excerpts of ambulatory ECG recordings from 79 subjects. The subjects were 70 men aged 30 to 84, and 8 women aged 55 to 71
Each record is two hours in duration and contains two signals, each sampled at 250 samples per second with 12-bit resolution over a nominal 20 millivolt input range. The sample values were rescaled after digitization with reference to calibration signals in the original analog recordings, in order to obtain a uniform scale of 200 ADC units per millivolt for all signals.
* DataBase Link: https://physionet.org/static/published-projects/edb/european-st-t-database-1.0.0.zip

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Project Structure
The project is structured as follows:

    * Data Preprocessing: This module contains functions for preprocessing heart beat signals, including noise removal, normalization, data augmentation, and splitting data into training, testing, and validation sets.
    * Model Building: This module consists of code for building and training classification models, including RNN and LSTM models.
    * Model Evaluation: Here, the performance of trained models is evaluated using metrics such as accuracy and loss. Visualizations such as loss and accuracy graphs, confusion matrices, and classification reports are generated for analysis.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Pre-requirements 
 There are necessary pre-requirements:
    The project utilizes several Python libraries including:
* keras                 |        2.14.0    |   : for data preprocessing and evaluation 
* matplotlib             |       3.7.2    |    : for data visualization
* numpy                   |      1.24.3    |    : for numerical computations
* pandas                   |     2.1.1    |    : for data manipulation
* seaborn                    |   0.12.2    |    : for statistical visualization
* sklearn                     |  0.0.post11    |    : for data preprocessing and evaluation
* tensorflow                   | 2.14.0    |    : for building and training deep learning models
* wfdb                        |  4.1.2    |    : for handling waveform database files
* glob                        |  3.11.5    |    : for extracting on files
* pywt                        |  1.4.1     |    : for wavelet transformations

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Data Preprocessing

* In the data preprocessing stage, the heart beat signals are processed to remove noise, normalize the data, handle imbalanced data, and perform data augmentation to enhance the training dataset.
Here Functions That Built For This Model:

    * find_paths: This function takes a list of file paths as input and returns a list of paths for all records in the given database. It specifically looks for files with the extension ".atr" in the provided folder path.

    * split_records: This function splits all records into beats. It iterates over each record, reads the ECG signals and annotations, filters and normalizes the signals, and then segments them into individual beats. It returns arrays of beats, annotation symbols, a dictionary encoding the symbols, and the number of unique symbols.

    * make_df: This function creates a DataFrame from the beats and labels data. It takes beats and labels as input and concatenates them into a DataFrame with two columns: "beats" and "annotations".

    * filter_ecg: This function filters the ECG signals to remove noise. It applies wavelet transform to the signals, sets a threshold for noise removal, and reconstructs the denoised signals using wavelet inverse transform.

    * normalize_ecg: This function normalizes the ECG signals. It subtracts the mean and divides by the standard deviation of the signal to standardize it.

    * encoder: This function encodes the labels into integers during the training stage. It takes a list of labels as input, converts them into a set to get unique labels, assigns integer values to each unique label, and creates a dictionary mapping labels to integers.

    * decoder: This function decodes the integer-encoded labels back to their original string labels during the testing stage. It takes a list of encoded labels and a dictionary mapping integers to labels as input, and returns the original string labels.

    * filter_data: This function filters the data to handle imbalanced classes. It identifies the majority class (usually normal data) and filters out rows with labels corresponding to the majority class. It returns two DataFrames: one containing abnormal data and the other containing normal data.

    * frequency_transform: This function applies frequency domain transformations, specifically adding Gaussian noise to the signal in the Fourier domain.

    * increase_data: This function performs data augmentation for minority classes. It takes abnormal data as input, applies augmentation techniques (such as frequency transformation), and creates new augmented data. It then combines the augmented data with the original abnormal data, samples an equal number of normal data, and creates a final balanced dataset.

    * spliting_data: This function splits the data into training, testing, and validation sets. It takes the data DataFrame as input, extracts beats and labels, and splits them into features (beats) and labels. It then splits the data using the train_test_split function from scikit-learn.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## Model Building
Two types of models are built in this project: RNN and LSTM. These models are constructed using the Keras library with TensorFlow backend

* Here is the Functions :
    * RNN_model: This function defines and trains an RNN (Recurrent Neural Network) model. It constructs a sequential model using Keras, consisting of three SimpleRNN layers with dropout regularization to prevent overfitting. The model is compiled with a sparse categorical cross-entropy loss function and the Adam optimizer. It then fits the model to the training data and returns the model summary and training history.
    * LSTM_model: Similar to the RNN_model function, this function defines and trains an LSTM (Long Short-Term Memory) model. It constructs a sequential model with three Bidirectional LSTM layers and dropout regularization. The model is compiled with a sparse categorical cross-entropy loss function and the Adam optimizer. It then fits the model to the training data and returns the model summary and training history.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Model Evaluation
The performance of the trained models is evaluated using various metrics such as accuracy, loss, precision, recall, and F1-score. Visualizations including loss and accuracy graphs, confusion matrices, and classification reports are generated to analyze the results.

* Here is the Functions :
    * evaluate_model: This function evaluates the performance of a trained model on the test data. It takes the trained model, test features (X_test), and test labels (Y_test) as input, and computes the test loss and accuracy using the evaluate method of the model.
  
    * predict: This function makes predictions using a trained model on the test data. It takes the test features (X_test), the trained model, and a dictionary for label decoding as input. It uses the trained model to predict labels for the test data, decodes the predicted labels using the provided dictionary, and returns the decoded labels.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Results and Analysis
The results of model evaluation are presented and analyzed to understand the effectiveness of different classification models in categorizing heart beat signals.

* Here is the Functions :
    * Loss_comp_graph: This function plots the training and validation loss curves for two different models. It takes the training history (containing loss and validation loss) for both models, as well as their names, and plots the loss curves using Matplotlib.
  
    * Accuracy_comp_graph: Similar to Loss_comp_graph, this function plots the training and validation accuracy curves for two different models. It takes the training history (containing accuracy and validation accuracy) for both models, as well as their names, and plots the accuracy curves using Matplotlib.

  
    * conf_mat: This function generates and plots a confusion matrix for the true and predicted labels. It takes the true labels (Y_test) and predicted labels (Y_pred) as input, along with an optional dictionary for label decoding. It uses Seaborn to plot the confusion matrix heatmap.
 


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Code Guide
Here The Steps of Code Running:

    *The code will execute the functions defined in the script.
    *It will read ECG data files from the specified folder and process them.
    *Preprocessing steps include filtering, normalization, and data augmentation.
    *The data will be split into training, testing, and validation sets.
    *Two models (RNN and LSTM) will be trained on the data.
    *Model performance (loss and accuracy) will be evaluated and displayed.
    *Predictions will be made on the test data using the trained models.
    *Statistical representations such as confusion matrices and classification reports will be generated and displayed.


