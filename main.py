"""
Main script to train the model and save the results.

This script sets the seed for reproducibility, trains the model using the `train` function from `train.py`,
and saves the actual and predicted labels to a CSV file.
"""

import json
import torch
import random
import numpy as np
import pandas as pd
from train import train


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

    # Train the model and get the true and predicted labels
    y_true, y_pred = train(
        n_classes=config['n_classes'],
        batch_size=config['batch_size'],
        b1=config['b1'],
        b2=config['b2'],
        n_epochs=config['n_epochs'],
        lr=config['lr'],
        behavioral_features=config['behavioral_features'],
        train_folder_path=config['train_folder_path'],
        test_folder_path=config['test_folder_path'],
        label_file=config['label_file'],
        milestones=config['milestones'],
        gamma=config['gamma'],
        patience=config['patience'],
        sampling_frequency=config['sampling_frequency'],
        weight_decay=config['weight_decay'],
        freq_min=config['freq_min'],
        freq_max=config['freq_max'],
        tensor_height=config['tensor_height']
    )
    
    # Create a DataFrame with true and predicted labels
    results_df = pd.DataFrame({
        'Actual Label': y_true.cpu().numpy(),
        'Predicted Label': y_pred.cpu().numpy()
    })

    # Save the results to a CSV file
    results_df.to_csv('test_results.csv', index=False)
    print('\nResults saved to test_results.csv')
