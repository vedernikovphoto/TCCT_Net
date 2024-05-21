import torch
from torch import nn

from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True


class TF_ConvModule(nn.Module):
    """
    Module for extracting temporal-frequency features.

    Parameters:
        output_features (int): Number of features in the output vector.

    Attributes:
        shallownet (nn.Sequential): Convolutional layers for initial feature extraction with specific
                                    adjustments for temporal and spatial convolutions.
        global_avg_pool (nn.AdaptiveAvgPool2d): Global average pooling layer to reduce spatial dimensions
                                                to a single value per feature map.
        reduce_dim (nn.Linear): Linear layer to reduce the number of features to the desired output size.
    """

    def __init__(self, output_features=64):
        """
        Initialize the TF_ConvModule.

        Parameters:
            output_features (int): Number of features in the output vector.
        """
        super(TF_ConvModule, self).__init__()

        # shallownet-like architecture for initial feature extraction from time-frequency data
        self.shallownet = nn.Sequential(
            nn.Conv2d(2, 16, (1, 10), padding=(0, 2)),
            nn.Conv2d(16, 32, (2, 1), padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 20), stride=(1, 15)),  # Temporal slicing
            nn.Dropout(0.6),
        )
        
        # Global average pooling to reduce spatial dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dimensionality reduction to desired feature size
        self.reduce_dim = nn.Linear(32, output_features)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TF_ConvModule.

        Parameters:
            x (torch.Tensor): Input tensor, expected to be a batch of 2D tensors with 
                              shape [batch_size, channels, height, width].

        Returns:
            torch.Tensor: Output tensor of reduced dimensionality with shape [batch_size, output_features].
        """

        x = self.shallownet(x)  
        x = self.global_avg_pool(x) 
        x = torch.flatten(x, 1) 
        x = self.reduce_dim(x)  
        
        return x
