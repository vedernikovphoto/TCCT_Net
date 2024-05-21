import numpy as np
import torch
from data.data_processing import batch_cwt


def evaluate(model, test_data, test_label, criterion_cls, freq_min, freq_max, tensor_height, sampling_frequency):
    """
    Evaluate the model on the test dataset.
    
    Args:
        model (nn.Module): The neural network model.
        test_data (torch.Tensor): Test dataset features.
        test_label (torch.Tensor): Test dataset labels.
        criterion_cls (nn.Module): The loss function used for evaluation.
        freq_min (float): Minimum frequency for CWT.
        freq_max (float): Maximum frequency for CWT.
        tensor_height (int): Number of discrete frequencies for CWT.
        sampling_frequency (int): Sampling frequency of the signal.

    Returns:
        float: Test accuracy.
        float: Test loss.
        torch.Tensor: Predicted labels.
    """
    
    model.eval()
    
    frequencies = np.linspace(freq_min, freq_max, tensor_height)
    cwt_representations_test = batch_cwt(test_data, frequencies, sampling_frequency=sampling_frequency)
    
    with torch.no_grad():
        Cls = model(test_data, cwt_representations_test)
        loss_test = criterion_cls(Cls, test_label)
        
        # Prediction and accuracy
        y_pred = torch.max(Cls, 1)[1]
        acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
    
    return acc, loss_test.item(), y_pred
