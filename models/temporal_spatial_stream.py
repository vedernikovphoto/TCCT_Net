import math
import torch.nn.functional as F

import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True


class TS_ConvModule(nn.Module):
    """
    A convolutional module for processing temporal-spatial data and preparing it for Transformer processing.

    Attributes:
        shallownet (nn.Sequential): A sequential container of layers consisting of convolutions,
                                    batch normalization, activation, pooling, and dropout.
        projection (nn.Sequential): A convolutional layer followed by a rearrangement layer to
                                    shape the output into the desired format for Transformer processing.
    """

    def __init__(self):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (2, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, 40, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TS_ConvModule module.

        Parameters:
            x (torch.Tensor): The input tensor with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The output tensor with embeddings re-arranged into a sequence format.
        """

        x = self.shallownet(x)
        x = self.projection(x)
        return x


class TS_AttentionModule(nn.Module):
    """
    TS_AttentionModule for a transformer.

    Parameters:
        emb_size (int): Size of each embedding vector.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate for attention weights.

    Attributes:
        keys (nn.Linear): Linear layer to create key vectors.
        queries (nn.Linear): Linear layer to create query vectors.
        values (nn.Linear): Linear layer to create value vectors.
        att_drop (nn.Dropout): Dropout layer for attention.
        projection (nn.Linear): Final projection layer.
    """

    def __init__(self, emb_size, num_heads, dropout):
        """
        Initialize the TS_AttentionModule.

        Parameters:
            emb_size (int): Size of each embedding vector.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate for attention weights.
        """

        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the TS_AttentionModule.

        Parameters:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor, optional): Mask to apply on attention weights.

        Returns:
            torch.Tensor: Output tensor after attention and linear transformation.
        """
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** 0.5
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class TS_ResidualAdd(nn.Module):
    """
    A residual connection followed by a layer norm.
    Wraps a function module to add the output of the function to its input, followed by normalization.

    Attributes:
        fn (nn.Module): Module to which the residual connection should be applied.
    """
    def __init__(self, fn):
        """
        Initialize the TS_ResidualAdd module.

        Parameters:
            fn (nn.Module): Module to which the residual connection should be applied.
        """

        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass for residual connection.
        
        Parameters:
            x (torch.Tensor): Input tensor to which residual is added.

        Returns:
            torch.Tensor: Output tensor after adding residual and applying module.
        """
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class TS_FeedForwardBlock(nn.Sequential):
    """
    A feedforward neural network block.

    Parameters:
        emb_size (int): Dimension of input and output tensors.
        expansion (int): Factor to expand the intermediate layer.
        drop_p (float): Dropout probability.
    """
    def __init__(self, emb_size, expansion, drop_p):
        """
        Initialize the TS_FeedForwardBlock.

        Parameters:
            emb_size (int): Dimension of input and output tensors.
            expansion (int): Factor to expand the intermediate layer.
            drop_p (float): Dropout probability.
        """        
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
        
        
class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply the GELU activation function to the input tensor.

        Parameters:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying GELU.
        """
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))
    
    
class TS_TransformerEncoderBlock(nn.Sequential):
    """
    Transformer Encoder Block that combines TS_AttentionModule and TS_FeedForwardBlock with residual connections.

    Parameters:
        emb_size (int): Embedding size.
        num_heads (int): Number of attention heads.
        drop_p (float): Dropout rate in attention and feed-forward layers.
        forward_expansion (int): Expansion factor for the feed-forward block.
        forward_drop_p (float): Dropout rate in the feed-forward block.
    """
    def __init__(self, emb_size, num_heads=10, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        """
        Initialize the TS_TransformerEncoderBlock.

        Parameters:
            emb_size (int): Embedding size.
            num_heads (int): Number of attention heads.
            drop_p (float): Dropout rate in attention and feed-forward layers.
            forward_expansion (int): Expansion factor for the feed-forward block.
            forward_drop_p (float): Dropout rate in the feed-forward block.
        """
        super().__init__(
            TS_ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                TS_AttentionModule(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            TS_ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                TS_FeedForwardBlock(emb_size, forward_expansion, forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )


class TS_TransformerEncoder(nn.Sequential):
    """
    Sequential container of TS_TransformerEncoderBlocks.

    Parameters:
        depth (int): Number of sequential Transformer Encoder Blocks.
        emb_size (int): Embedding size used throughout the transformer.
    """
    def __init__(self, depth, emb_size):
        """
        Initialize the TS_TransformerEncoder.

        Parameters:
            depth (int): Number of sequential Transformer Encoder Blocks.
            emb_size (int): Embedding size used throughout the transformer.
        """        
        super().__init__(*[TS_TransformerEncoderBlock(emb_size) for _ in range(depth)])


class TS_Stream(nn.Module):
    """
    Module for processing temporal-spatial data using convolution and transformer encoder blocks.

    Parameters:
        depth (int): Number of layers in the TS_TransformerEncoder.
        emb_size (int): Dimensionality of the embedding space.

    Attributes:
        TS_ConvModule (TS_ConvModule): The TS_ConvModule module used for initial data processing.
        TS_TransformerEncoder (TS_TransformerEncoder): Transformer encoder composed of multiple layers.
    """
    def __init__(self, depth=5, emb_size=40):
        """
        Initialize the TS_Stream.

        Parameters:
            depth (int): Number of layers in the TS_TransformerEncoder.
            emb_size (int): Dimensionality of the embedding space.
        """
        super().__init__()
        self.TS_ConvModule = TS_ConvModule()
        self.TS_TransformerEncoder = TS_TransformerEncoder(depth, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TS_Stream.

        Parameters:
            x (torch.Tensor): Input tensor to be processed.

        Returns:
            torch.Tensor: Output tensor after processing through TS_ConvModule and TS_TransformerEncoder.
        """
        x = self.TS_ConvModule(x)
        x = self.TS_TransformerEncoder(x)
        return x
