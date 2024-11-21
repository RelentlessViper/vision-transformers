# Import necessary dependencies
import torch
from torch import nn
from const import *

from patch_embedding import PatchEmbedding
from encoder import Encoder

class VIT(nn.Module):
    """
    Vision Transformer (VIT) implementation
    
    Attributes
    ----------
    `num_layers`: int = 12
        The amount of Encoder blocks stacked.
    `latent_size`: int = 768
        The size of image embeddings.
    `num_heads`: int = 12
        The amount of heads used in the Attention mechanism.
    `num_classes`: int = 10
        The amount of classes in the dataset.
    `dropout`: float = 0.1
        The probability of dropping each value from the layer's output.
    `device`: str = 'cpu'
        The device. Can be either 'cpu', or 'cuda:n', where 'n' is the number of GPU processors.
    """
    
    def __init__(
        self,
        num_layers: int = NUM_ENCODERS,
        latent_size: int = LATENT_SIZE,
        num_heads: int = NUM_HEADS,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
        device: str = 'cpu'
    ) -> None:
        """
        Initialize Vision Transformer (VIT).
        
        Parameters
        ----------
        `num_layers`: int = 12
            The amount of Encoder blocks stacked.
        `latent_size`: int = 768
            The size of image embeddings.
        `num_heads`: int = 12
            The amount of heads used in the Attention mechanism.
        `num_classes`: int = 10
            The amount of classes in the dataset.
        `dropout`: float = 0.1
            The probability of dropping each value from the layer's output.
        `device`: str = 'cpu'
            The device. Can be either 'cpu', or 'cuda:n', where 'n' is the number of GPU processors.
            
        Returns
        ----------
        `self`: VIT
            VIT class object.
        """
        
        super(VIT, self).__init__()
        
        self.num_layers = num_layers
        self.latent_size = latent_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device
        
        self.patch_embedding = PatchEmbedding()
        
        self.encoder_stack = nn.ModuleList([Encoder(self.latent_size, self.num_heads, self.dropout, self.device) for _ in range(self.num_layers)])
        
        self.MLP = nn.Sequential(
            nn.LayerNorm(self.latent_size),
            nn.Linear(self.latent_size, self.latent_size),
            nn.Linear(self.latent_size, self.num_classes)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Get the output from VIT by passing the input through the list of Encoders.
        
        Parameters
        ----------
        `input`: torch.Tensor
            Input image embeddings
        
        Returns
        ----------
        `result`: torch.Tensor
            The tensor containing the probabilities for each class.
        """
        
        x = self.patch_embedding(input)
        
        for encoder_layer in self.encoder_stack:
            x = encoder_layer(x)
        
        class_embedding = x[:, 0]
        
        result = self.MLP(class_embedding)
        
        return result