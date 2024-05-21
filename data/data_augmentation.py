import torch
import numpy as np


def interaug(data, label, batch_size, signal_length, device, num_behavioral_features):
    """
    Perform data augmentation by Segmentation and Recombination (S&R) technique.

    Args:
        data (np.array): Data to be augmented.
        label (np.array): Labels corresponding to the input data.
        batch_size (int): Size of the batch.
        signal_length (int): Length of the signal.
        device (torch.device): The device (CPU/GPU) to transfer the tensors.
        num_behavioral_features (int): Number of behavioral features in the data.

    Returns:
        torch.Tensor: Augmented data tensor.
        torch.Tensor: Augmented labels' tensor.
    """

    aug_data, aug_label = [], []
    
    # Define the number of segments, the length of each segment,
    # the number of augmented samples per class based on the batch size
    n_segments = 30
    segment_length = signal_length // n_segments
    total_samples_per_class = batch_size // len(np.unique(label))

    # Iterate over each unique class
    for cls4aug in np.unique(label):
        # Find indices where current class label exists and extract the corresponding data for this class
        cls_idx = np.where(label == cls4aug)
        tmp_data = data[cls_idx]

        # Determine the number of samples to augment to match the total_samples_per_class
        n_samples_needed = total_samples_per_class
        # Initialize a temporary array for augmented data of this class
        tmp_aug_data = np.zeros((n_samples_needed, 1, num_behavioral_features, signal_length))

        # Perform augmentation 
        for ri in range(n_samples_needed):
            for rj in range(n_segments):
                # Select random indices within the class data for each segment
                rand_idx = np.random.randint(0, len(tmp_data), n_segments)
                start = rj * segment_length
                end = (rj + 1) * segment_length
                
                # Replace the segment in the augmented data with the randomly selected segment
                tmp_aug_data[ri, :, :, start:end] = tmp_data[rand_idx[rj], :, :, start:end]

        aug_data.append(tmp_aug_data)
        aug_label.append(np.full(n_samples_needed, cls4aug))

    # Concatenate all augmented data and labels, then shuffle
    aug_data = np.concatenate(aug_data)
    aug_label = np.concatenate(aug_label)
    aug_shuffle = np.random.permutation(len(aug_data))
    aug_data = aug_data[aug_shuffle, :, :]
    aug_label = aug_label[aug_shuffle]

    # Convert numpy arrays to PyTorch tensors and transfer to the specified device
    aug_data = torch.from_numpy(aug_data).to(device).float()
    aug_label = torch.from_numpy(aug_label).to(device).long()

    return aug_data, aug_label
