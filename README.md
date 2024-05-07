# Heart-beats-classifing
    This Python project focuses on classifying heartbeat signals using various deep learning models. It involves preprocessing the data, building and training models, and evaluating their performance.

# Table of Contents
* DataBase
* Libraries
* Functions
    * Preprocessing
    * Models
    * Evaluation
    * Statistical Graphs
* Reading Files & Applying Functions
* Results
* Conclusion

## DataBase
This database consists of 90 annotated excerpts of ambulatory ECG recordings from 79 subjects. The subjects were 70 men aged 30 to 84, and 8 women aged 55 to 71. (Information is missing for one subject.
Each record is two hours in duration and contains two signals, each sampled at 250 samples per second with 12-bit resolution over a nominal 20 millivolt input range. The sample values were rescaled after digitization with reference to calibration signals in the original analog recordings, in order to obtain a uniform scale of 200 ADC units per millivolt for all signals.
DataBase Link: https://physionet.org/static/published-projects/edb/european-st-t-database-1.0.0.zip
## Libraries
  The project utilizes several Python libraries including:
* wfdb: for handling waveform database files
* pandas: for data manipulation
* matplotlib: for data visualization
* numpy: for numerical computations
* tensorflow and keras: for building and training deep learning models
* pywt: for wavelet transformations
* scikit-learn: for data preprocessing and evaluation
* seaborn: for statistical visualization
## Functions
### Preprocessing
1. Collecting Data
1.1 find_paths: Finds paths for all records in the database.
1.2 split_records: Splits records into beats. 
1.3 make_df: Collects data in CSV format for handling imbalanced data.
2. Data Manipulation
2.1 filter_ecg: Removes noise from ECG signals.
2.2 normalize_ecg: Normalizes ECG signals.
2.3 filter_data: Filters data to handle imbalanced classes.
2.4 increase_data: Performs data augmentation for minority classes.

### Models
3.1 RNN_model: Builds and trains an RNN model for heartbeat classification.
3.2 LSTM_model: Builds and trains an LSTM model for heartbeat classification.

### Evaluation
4.1 evaluate_model: Evaluates the performance of a trained model.
4.2 predict: Makes predictions using a trained model.
4.3 conf_mat: Generates and visualizes the confusion matrix.

### Statistical Graphs
5.1 Loss_comp_graph: Compares the loss of multiple models over epochs.
5.2 Accuracy_comp_graph: Compares the accuracy of multiple models over epochs.
Reading Files & Applying Functions
Reads files from a specified folder path and applies preprocessing functions to prepare the data for training.
Results
Trains various models including RNN and LSTM.
Evaluates the performance of trained models using test data.
Generates statistical graphs to compare model performance.
Conclusion
Summarizes the findings and performance of different models.
Saves the trained models and processed data for future use.
License
Specify the license under which your project is released.
