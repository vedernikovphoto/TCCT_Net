import json
import torch
import random
import numpy as np
import pandas as pd
from torch import nn

from models.feature_fusion import Decision_Fusion
from test import evaluate
from data.data_loader import get_source_data_inference

gpus = [0]


def inference(n_classes, behavioral_features, target_mean, target_std, inference_folder_path, label_file_inference,
              sampling_frequency, freq_min, freq_max, tensor_height, model_weights_path):
    """
    Load the pre-trained model weights and evaluate it on the inference data.

    Args:
        n_classes (int): Number of classes for the classification task.
        behavioral_features (list of str): List of behavioral features used.
        target_mean (float): Mean value of the training data used for standardization.
        target_std (float): Standard deviation of the training data used for standardization.
        inference_folder_path (str): Path to the folder containing the inference data.
        label_file_inference (str): Path to the CSV file containing the labels.
        sampling_frequency (int): Sampling frequency of the signal.
        freq_min (float): Minimum frequency for CWT.
        freq_max (float): Maximum frequency for CWT.
        tensor_height (int): Number of discrete frequencies for CWT.
        model_weights_path (str): Path to the pre-trained model weights.

    Returns:
        tuple: Tuple containing the inference labels and predicted labels.
    """

    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model and load pre-trained weights
    model = Decision_Fusion(n_classes)
    model = nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()

    # Load the data from the inference directory
    inference_signal_data, inference_label = get_source_data_inference(
        inference_folder_path, label_file_inference, behavioral_features, target_mean, target_std)

    # Convert the data into PyTorch tensors and transfer to the appropriate device
    inference_signal_data = torch.from_numpy(inference_signal_data).float().to(device)
    inference_label = torch.from_numpy(inference_label).long().to(device)

    # Initialize the loss function
    criterion_cls = torch.nn.CrossEntropyLoss().to(device)

    # Evaluate the model on the inference data
    inference_acc, loss_inference, y_pred = evaluate(model, inference_signal_data, inference_label, criterion_cls,
                                                     freq_min, freq_max, tensor_height, sampling_frequency)

    print(f'Inference Accuracy: {inference_acc}')
    print(f'Inference Loss: {loss_inference}')

    return inference_label, y_pred


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


# Setting the seed for reproducibility
seed_n = 42
random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)
torch.cuda.manual_seed_all(seed_n)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # Load the configuration
    config = load_config('config.json')

    # Run inference to get the true and predicted labels
    y_true, y_pred = inference(
        n_classes=config['n_classes'],
        behavioral_features=config['behavioral_features'],
        target_mean=config['target_mean'],
        target_std=config['target_std'],
        inference_folder_path=config['inference_folder_path'],
        label_file_inference=config['label_file_inference'],
        sampling_frequency=config['sampling_frequency'],
        freq_min=config['freq_min'],
        freq_max=config['freq_max'],
        tensor_height=config['tensor_height'],
        model_weights_path=config['final_model_weights']
    )

    # Create a DataFrame with true and predicted labels
    results_df = pd.DataFrame({
        'Actual Label': y_true.cpu().numpy(),
        'Predicted Label': y_pred.cpu().numpy()
    })

    # Save the results to a CSV file
    results_df.to_csv('inference_results.csv', index=False)
    print('\nResults saved to inference_results.csv')
