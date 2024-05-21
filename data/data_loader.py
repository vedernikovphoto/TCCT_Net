import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_csv_data(folder_path, label_file, behavioral_features):
    """
    Load data from CSV files in a folder and corresponding labels.

    Args:
        folder_path (str): Path to the folder containing the CSV files.
        label_file (str): Path to the CSV file containing the labels.
        behavioral_features (list of str): List of behavioral features to extract from the data.

    Returns:
        np.array: Array of data extracted from the CSV files.
        np.array: Array of labels corresponding to the data.
    """
        
    labels_df = pd.read_csv(label_file)
    all_data, all_labels = [], []

    # Process each CSV file in the directory
    for filename in tqdm(os.listdir(folder_path), desc="Loading data"):
        if filename.endswith('.csv'):
            subject_id = filename.split('.')[0]
            subject_file = os.path.join(folder_path, filename)
            subject_data = pd.read_csv(subject_file)

            # Stack data for selected behavioral features
            subject_data_values = np.stack([subject_data[col].values for col in behavioral_features], axis=0)
            subject_label = labels_df[labels_df['chunk'].str.contains(subject_id)]['label'].values

            # Append data and label if label exists
            if len(subject_label) > 0:
                all_data.append(subject_data_values)
                all_labels.append(subject_label[0])
            else:
                print(f"No label found for subject {subject_id}")

    # Reshape the collected data and convert it to numpy arrays
    all_data = np.array(all_data)
    all_data = np.expand_dims(all_data, axis=1)  
    all_labels = np.array(all_labels)

    return all_data, all_labels


def get_source_data(train_folder_path, test_folder_path, label_file, behavioral_features):
    """
    Load and preprocess training and testing data from the specified folders.

    Args:
        train_folder_path (str): Path to the folder containing the training data.
        test_folder_path (str): Path to the folder containing the testing data.
        label_file (str): Path to the CSV file containing the labels.
        behavioral_features (list of str): List of behavioral features to extract from the data.

    Returns:
        np.array: Processed training data.
        np.array: Labels for the training data.
        np.array: Processed testing data.
        np.array: Labels for the testing data.
    """

    # Load training data
    print('\nLoading train data ...')
    train_data, train_labels = load_csv_data(train_folder_path, label_file, behavioral_features)
    train_labels = train_labels.reshape(1, -1)

    # Shuffle the training data
    shuffle_index = np.random.permutation(len(train_data))
    train_data = train_data[shuffle_index, :, :, :]
    train_labels = train_labels[0][shuffle_index]

    # Load testing data
    print('\nLoading test data ...')
    test_data, test_labels = load_csv_data(test_folder_path, label_file, behavioral_features)
    test_labels = test_labels.reshape(-1)  

    # Standardize both train and test data using training data statistics
    target_mean = np.mean(train_data)
    target_std = np.std(train_data)
    train_data = (train_data - target_mean) / target_std
    test_data = (test_data - target_mean) / target_std

    return train_data, train_labels, test_data, test_labels


def get_source_data_inference(inference_folder_path, label_file_inference,
                              behavioral_features, target_mean, target_std):
    """
    Load and preprocess inference data from the specified folder.

    Args:
        inference_folder_path (str): Path to the folder containing the inference data.
        label_file_inference (str): Path to the CSV file containing the labels.
        behavioral_features (list of str): List of behavioral features to extract from the data.
        target_mean (float): Mean value of the training data used for standardization.
        target_std (float): Standard deviation of the training data used for standardization.

    Returns:
        np.array: Processed testing data.
        np.array: Labels for the testing data.
    """

    # Load inference data
    print('\nLoading data for inference ...')
    inference_data, inference_labels = load_csv_data(inference_folder_path, label_file_inference, behavioral_features)
    inference_labels = inference_labels.reshape(-1)

    # Standardize inference data using provided training data statistics
    inference_data = (inference_data - target_mean) / target_std

    return inference_data, inference_labels
