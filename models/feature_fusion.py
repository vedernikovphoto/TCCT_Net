from models.temporal_frequency_stream import TF_ConvModule
from models.temporal_spatial_stream import TS_Stream

import torch
from torch import nn

from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True


class Decision_Fusion(nn.Module):
    """
    Module for decision fusion of Temporal-Spatial and Temporal-Frequency streams' outputs.

    Parameters:
        n_classes (int): Number of classes for the classification task.

    Attributes:
        TS_Stream (TS_Stream): Temporal-Spatial feature extraction stream.
        TF_ConvModule (TF_ConvModule): Temporal-Frequency feature extraction stream.
        classification_head_TS (nn.Sequential): Classification head in Temporal-Spatial stream.
        classification_head_TF (nn.Sequential): Classification head in Temporal-Frequency stream.
        TS_weight (nn.Parameter): Learnable weight for the Temporal-Spatial stream classifier output.
        TF_weight (nn.Parameter): Learnable weight for the Temporal-Frequency stream classifier output.
    """
    
    def __init__(self, n_classes):
        super(Decision_Fusion, self).__init__()

        # Model components for each modality
        self.TS_Stream = TS_Stream()
        self.TF_ConvModule = TF_ConvModule()

        # Classification heads for each modality
        self.classification_head_TS = nn.Sequential(
            nn.Linear(520, 128),  
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
        
        self.classification_head_TF = nn.Sequential(
            nn.Linear(64, 32),  
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(32, n_classes)
        )

        # Weights for combining modalities
        self.TS_weight = nn.Parameter(torch.tensor(0.5)) 
        self.TF_weight = nn.Parameter(torch.tensor(4.0))

    def forward(self, TS_input, TF_input):
        """
        Forward pass for the Decision_Fusion.

        Parameters:
            TS_input (torch.Tensor): Input tensor for the Temporal-Spatial stream.
            TF_input (torch.Tensor): Input tensor for the Temporal-Frequency stream.

        Returns:
            torch.Tensor: Combined classification predictions from both streams.
        """
        # Process inputs through each modality's pathway
        TS_output = self.TS_Stream(TS_input)
        TF_output = self.TF_ConvModule(TF_input)

        # Flatten outputs for classification
        TS_output = TS_output.view(TS_output.size(0), -1)
        TF_output = TF_output.view(TF_output.size(0), -1)

        # Get predictions for each modality
        TS_preds = self.classification_head_TS(TS_output)
        TF_preds = self.classification_head_TF(TF_output)

        # Combine predictions using learned weights
        combined_preds = self.TS_weight * TS_preds + self.TF_weight * TF_preds

        return combined_preds
