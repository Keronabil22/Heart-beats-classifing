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
        * Parameters:
            This function does not take any parameters directly but relies on a globally defined variable folder_path.
        * Function Details:
            * Search for .atr Files:
                Use the glob module to find all files with the .atr extension in the specified folder_path.
            * Process File Paths:
                * Convert the file paths to a uniform format by:
                    * Removing the file extension using os.path.splitext.
                    * Replacing backslashes (\\) with forward slashes (/) to ensure compatibility across different operating systems.
            * Return Processed Paths:
                    Return the list of processed file paths.

    * split_records: This function splits all records into beats. It iterates over each record, reads the ECG signals and annotations, filters and normalizes the signals, and then segments them into individual beats. It returns arrays of beats, annotation symbols, a dictionary encoding the symbols, and the number of unique symbols.
        * Parameters:
            * record_list (list): List of ECG record names to process.
        * Function Details:
            * Initialize Lists:
                * beats: A list to store the extracted ECG beat segments.
                * annotation_symbols: A list to store the annotations corresponding to each beat.
            * Iterate Through Each Record:
                * For each record name in record_list:
                * Read the ECG record using wfdb.rdrecord(record_Name).
                * Check if the signal names contain 'V4' or 'V5', indicating the lead used for the ECG signal.
            * Extract ECG Signal:
                * If the signal contains 'V4' or 'V5':
                * Extract the ECG signal for the corresponding lead.
                * Apply wavelet denoising, Wiener filtering, and normalization to the ECG signal using wavelet_denoising, wiener, and normalize_ecg functions, respectively.
            * Fetch Annotations:
                * Read the annotations from the record using wfdb.rdann(record_Name, 'atr').
                * Extract the indices of QRS complexes from the annotations.
            * Extract Beat Segments:
                * Define the beat duration (assumed to be 250 ms).
                * For each QRS complex index:
                    * Calculate the start and end indices of the beat segment around the QRS complex.
                    * Extract the beat segment from the ECG signal.
                    * If the beat length is less than 250 samples, pad the beat with the last sample to ensure a consistent length.
                    * Append the beat segment to the beats list.
                * Append the annotation symbols to the annotation_symbols list.
            * Post-process Annotations:
                * Flatten the list of annotation symbols.
                * Encode the annotation symbols using the encoder function, which returns the encoded symbols, a dictionary mapping symbols to their encodings, and the number of unique symbols.
            * Create DataFrame:
                * Create a DataFrame from the beats and encoded annotation symbols using the make_df function.
            * Return Values:
                * beats: Array of processed ECG beat segments.
                * annotation_symbols: Array of encoded annotation symbols.
                * annotation_symbols_dic: Dictionary mapping annotation symbols to their encodings.
                * symb_num: Number of unique annotation symbols.

    * make_df: This function creates a DataFrame from the beats and labels data. It takes beats and labels as input and concatenates them into a DataFrame with two columns: "beats" and "annotations".
        * Parameters:
            * beats (list): A list of ECG beat segments.
            * Labels (list): A list of annotations corresponding to each beat segment.
        * Function Details:
            * Convert Lists to pandas Series:
                * Convert the list of ECG beats to a pandas Series named beats.
                * Convert the list of labels to a pandas Series named annotations.
            * Create DataFrame:
                * Concatenate the two Series along the columns (axis=1) to form a DataFrame with two columns: beats and annotations.
            * Shuffle DataFrame:
                * Shuffle the DataFrame randomly using sample(frac=1). This ensures that the rows are in a random order.
                * Reset the index of the shuffled DataFrame to ensure a clean, continuous index without any gaps using reset_index(drop=True).
            * Return the DataFrame:
                * Return the shuffled DataFrame.

    * wavelet_denoising: This function filters the ECG signals to remove noise. It applies wavelet transform to the signals, sets a threshold for noise removal, and reconstructs the denoised signals using wavelet inverse transform.
        * Parameters:
            * ecg_signal (array-like): The raw ECG signal that needs to be denoised.
        * Function Details:
            * Apply Wavelet Transform:
                * Choose the wavelet type (in this case, 'db6' which is Daubechies 6 wavelet).
                * Decompose the ECG signal into wavelet coefficients using pywt.wavedec.
            * Set a Threshold for Noise Removal:
                * Define a threshold value (0.5 in this example) to distinguish between noise and signal components. This value may need adjustment based on the characteristics of the ECG signal.
            * Apply Thresholding to Remove Noise:
                * Use soft thresholding on the wavelet coefficients to suppress noise. This is done using pywt.threshold which applies the threshold to each set of coefficients.
            * Reconstruct the Denoised Signal:
                * Reconstruct the signal from the thresholded wavelet coefficients using pywt.waverec.
            * Return the Denoised Signal:
                * Return the reconstructed, denoised ECG signal.
    * normalize_ecg: This function normalizes the ECG signals. It subtracts the mean and divides by the standard deviation of the signal to standardize it.
        * Parameters:
            * ecg_signal (array-like): The input ECG signal to be normalized.
        * Function Details:
            * Calculate Mean and Standard Deviation:
                * Compute the mean of the ECG signal using np.mean(ecg_signal).
                * Compute the standard deviation of the ECG signal using np.std(ecg_signal).
            * Normalize the Signal:
                * Subtract the mean from each element of the ECG signal.
                * Divide the result by the standard deviation to obtain the normalized signal.
            * Return the Normalized Signal:
                * Return the normalized ECG signal.
    * encoder: This function encodes the labels into integers during the training stage. It takes a list of labels as input, converts them into a set to get unique labels, assigns integer values to each unique label, and creates a dictionary mapping labels to integers.
        * Parameters:
            * labels (list): A list of string labels to be encoded.
        * Function Details:
            * Create a Set of Unique Labels:
                * Convert the list of labels into a set to obtain unique labels.
            * Generate a Set of Integers:
                * Create a set of integers ranging from 0 to the number of unique labels minus one.
            * Map Labels to Integers:
                * Create a dictionary that maps each unique string label to a corresponding integer using dict(zip(my_set_labels, my_set_numbers)).
            * Convert Labels to Integers:
                * Transform the original list of string labels into a list of integers using the created dictionary. If a label is not found in the dictionary, it remains unchanged (though this case should not occur given the initial mapping).
            * Return Values:
                * converted_list: The list of integer-encoded labels.
                * my_dic: The dictionary mapping string labels to integers.
                * len(set(converted_list)): The number of unique labels.
    * decoder: This function decodes the integer-encoded labels back to their original string labels during the testing stage. It takes a list of encoded labels and a dictionary mapping integers to labels as input, and returns the original string labels.
        * Parameters:
            * labels (list): A list of integer-encoded labels.
            * dic (dict): A dictionary mapping integer labels to their corresponding string representations.
        * Function Details:
            * Decode Labels:
                * For each integer label in the input list labels, find the corresponding key (string label) in the dictionary dic where the value matches the integer label.
        * Return a list of decoded string labels.
    * filter_data: This function filters the data to handle imbalanced classes. It identifies the majority class (usually normal data) and filters out rows with labels corresponding to the majority class. It returns two DataFrames: one containing abnormal data and the other containing normal data.
        * Parameters:
            * data (DataFrame): The input DataFrame containing the data to be filtered.
        * Function Details:
            * Find the Majority Value:
                * Calculate the counts of unique values in the specified column (annotations).
                * Identify the majority value (the value with the highest count) using value_counts().idxmax().
            * Filter the DataFrame:
                * Create two separate DataFrames:
                    * abnormal_data: Contains rows with values different from the majority value (e.g., anomalies, outliers).
                    * normal_data: Contains rows with the majority value (e.g., normal instances).
                * Use boolean indexing to filter rows based on the majority value.
            * Return Filtered DataFrames:
                * Return the filtered abnormal_data and normal_data DataFrames.
    * frequency_transform: This function applies frequency domain transformations, specifically adding Gaussian noise to the signal in the Fourier domain.
        * Parameters:
            * signal (array-like): The input signal to be transformed.
            * noise_level (float, optional): The standard deviation of the Gaussian noise to be added. Default is 0.01.
        * Function Details:
            * Generate Gaussian Noise:
                * Create Gaussian noise with a mean of 0 and a standard deviation specified by noise_level.
                * The length of the noise array is the same as the input signal.
            * Apply Fast Fourier Transform (FFT):
                * Compute the FFT of the input signal using np.fft.fft.
            * Add Noise in Frequency Domain:
                * Add the Gaussian noise to the FFT-transformed signal.
            * Inverse Fast Fourier Transform (IFFT):
                * Apply the inverse FFT to the noisy frequency domain signal to transform it back to the time domain.
                * Use .real to get the real part of the transformed signal, discarding any imaginary components that may arise due to numerical errors.
            * Return the Transformed Signal:
                * Return the real part of the inverse FFT-transformed signal.
    * increase_data: This function performs data augmentation for minority classes. It takes abnormal data as input, applies augmentation techniques (such as frequency transformation), and creates new augmented data. It then combines the augmented data with the original abnormal data, samples an equal number of normal data, and creates a final balanced dataset.
        * Parameters:
            * abnormal_data (DataFrame): DataFrame containing the minority class (abnormal data).
            * normal_data (DataFrame): DataFrame containing the majority class (normal data).
        * Function Details:
            * Data Augmentation:
                * Iterate over each row in the abnormal_data DataFrame.
                * Extract the signal from the beats column of each row.
                * Apply augmentation techniques (e.g., frequency transformation) to the signal to generate augmented signals.
                * Append the augmented signals to separate lists (augmented_data1, augmented_data2).
            * Create Augmented DataFrames:
                * Create new DataFrames (augmented_df1, augmented_df2) from the augmented data, preserving the annotations.
            * Combine DataFrames:
                * Concatenate the original abnormal_data DataFrame with the augmented DataFrames (augmented_df1, augmented_df2).
            * Balance Classes:
                * Randomly sample the normal_data DataFrame to match the length of the augmented abnormal_data.
                * Concatenate the balanced normal_data DataFrame with the augmented abnormal_data.
            * Shuffle Data:
                * Shuffle the final DataFrame to ensure randomness in the order of samples.
            * Return Final Data:
                * Return the shuffled final DataFrame containing both augmented and original data.
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
        * Parameters:
            * Y_test (array-like): The true labels.
            * Y_pred (array-like): The predicted labels.
            * dic (dict, optional): A dictionary for encoding predicted labels if they are integers. Default is an empty dictionary.
        * Function Details:
            * Encode Predicted Labels (if necessary):
                * If the predicted labels (Y_pred) are integers, convert them using the provided dictionary (dic).
                * Iterate over Y_pred and use the dictionary to map each label to its encoded form, storing the result in Y_pred_encoded.
            * Compute Confusion Matrix:
                * Calculate the confusion matrix using confusion_matrix(Y_test, Y_pred) from sklearn.metrics.
            * Convert Tensor to NumPy Array (if necessary):
                * If the confusion matrix is a TensorFlow tensor, convert it to a NumPy array.
            * Initialize Metric Lists:
                * Create empty lists to store true positives, true negatives, false positives, false negatives, sensitivity, and specificity for each class.
            * Calculate Metrics for Each Class:
                * Determine the number of classes from the shape of the confusion matrix.
                * For each class i:
                    * Calculate TP, FP, FN, and TN.
                    * Append these values to their respective lists.
                    * Calculate sensitivity (tp / (tp + fn)) and specificity (tn / (tn + fp)), and append them to their respective lists.
                    * Print the metrics for each class.
            * Plot Confusion Matrix:
                * Visualize the confusion matrix using seaborn.heatmap.
            * Return Values:
                * Return the encoded or original predicted labels, and the lists of TP, TN, FP, and FN.
    * ROC_graph : The ROC_graph function calculates and plots the ROC curves for each class in a multi-class classification problem. ROC curves are a graphical representation of a classifier's performance, plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings.
        * Parameters:
            * y_true (array-like): True labels of the dataset.
            * y_pred_prob (array-like): Predicted probabilities for each class, typically obtained from a classifier.
            * n_classes (int): The number of classes in the classification problem.
        * Function Details:
            * Initialize dictionaries:
                * fpr, tpr, and roc_auc dictionaries are initialized to store the false positive rates, true positive rates, and area under the curve (AUC) values for each class, respectively.
            * Compute ROC curve and AUC for each class:
                * Loop through each class (from 0 to n_classes-1).
                * For each class i, convert the true labels to a binary format (where 1 indicates the presence of class i and 0 indicates the absence).
                * Use the roc_curve function to compute the FPR and TPR values.
                * Use the auc function to compute the AUC value for the ROC curve of class i.
            * Plot ROC curves:
                * Create a new figure for the plot with a specified size.
                * Loop through each class and plot the FPR and TPR values, labeling each curve with the class number and its corresponding AUC value.
                * Add a diagonal dashed line representing a random classifier's performance (where TPR equals FPR).
                * Set the limits for the x and y axes.
                * Label the axes and add a title.
                * Add a legend in the lower right corner to identify each class's ROC curve.
                * Display the plot.

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
    *Statistical representations such as Accuracy , Loss , ROC Graph ,confusion matrices and classification reports will be generated and displayed.


