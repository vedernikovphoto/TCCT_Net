import pywt
import numpy as np
import torch


def batch_cwt(batch_signals, frequencies, sampling_frequency):
    """
    Compute the Continuous Wavelet Transform (CWT) for a batch of signals.

    Args:
        batch_signals (torch.Tensor): Input batch of signals with shape [batch_size, channels, features, signal_length].
        frequencies (np.array): Array of frequencies to use for the CWT.
        sampling_frequency (int): Sampling frequency of the signals.

    Returns:
        torch.Tensor: Tensor containing the CWT coefficients for the input batch of signals.
    """
        
    # Extract the batch size and the number of features from the batch_signals shape
    batch_size, _, features, _ = batch_signals.shape
    cwt_batch = []

    # Process each signal in the batch to compute the CWT
    for i in range(batch_size):
        cwt_features = []
        for j in range(features):
            # Extract the signal, ensure it's on CPU and flattened to 1D
            signal = batch_signals[i, :, j, :].squeeze().cpu().detach().numpy()

            # Compute the CWT using the specified wavelet function and scaling
            coefficients, _ = pywt.cwt(signal, frequencies, 'cmor1.5-1.0', sampling_period=1/sampling_frequency)
            coefficients = np.abs(coefficients)

            # Collect CWT results for each feature
            cwt_features.append(coefficients)

        # Stack CWT results for all features of a single batch item
        cwt_features_stacked = np.stack(cwt_features, axis=0)
        cwt_batch.append(cwt_features_stacked)

    # Convert the list of numpy arrays to a single numpy array
    cwt_batch_np = np.array(cwt_batch)
    
    # Convert the numpy array to a torch tensor for compatibility with torch operations
    cwt_batch_tensor = torch.tensor(cwt_batch_np, dtype=torch.float)

    return cwt_batch_tensor
