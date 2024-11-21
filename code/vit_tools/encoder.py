# Import necessary dependencies
import torch
from torch import nn
from const import *

class Encoder(nn.Module):
    """
    Encoder class for VIT.
    
    Attributes
    ----------
    `latent_size`: int = 768
        The size of image embeddings.
    `num_heads`: int = 12
        The amount of heads used in the Attention mechanism.
    `dropout`: float = 0.1
        The probability of dropping each value from the layer's output.
    """
    
    def __init__(
        self,
        latent_size: int = LATENT_SIZE,
        num_heads: int = NUM_HEADS,
        dropout: float = DROPOUT
    ) -> None:
        """
        Initialize the Encoder class.
        
        Parameters
        ----------
        `latent_size`: int = 768
            The size of image embeddings.
        `num_heads`: int = 12
            The amount of heads used in the Attention mechanism.
        `dropout`: float = 0.1
            The probability of dropping each value from the layer's output.
        
        Returns
        ----------
        `self`: Encoder
            Encoder class object.
        """
        
        super(Encoder, self).__init__()
        
        self.latent_size = latent_size
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.layer_norm_0 = nn.LayerNorm(self.latent_size)
        self.layer_norm_1 = nn.LayerNorm(self.latent_size)
        
        self.multi_attn = nn.MultiheadAttention(
            embed_dim=self.latent_size,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        self.linear = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size * 4),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.latent_size * 4, self.latent_size),
            nn.Dropout(p=self.dropout)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Get the output from the Encoder.
        
        The Encoder architecture works in the following way (The shape remains the same throughout the process):
            1) Normalize the input using layer normalization;
            2) Get the output from the Multi-Head Attention layer (without Attention masks);
            3) Add the output from the previous layer and add it to the initial input (Residual connection #1);
            4) Normalize the data using layer normalization;
            5) Get the output from MLP layer;
            6) Residual connection #2.
        
        Proper explanation can be found [here](https://arxiv.org/pdf/2010.11929v2).
        
        Parameters
        ----------
        `input`: torch.Tensor()
            The batch of image embeddings.
        
        Returns
        ----------
        `residual_connection_1`: torch.Tensor()
            The result of Encoder block that has the same shape as the `input`.
        """
        
        input_normalized = self.layer_norm_0(input)
        attention_input_normalized, _ = self.multi_attn(input_normalized, input_normalized, input_normalized)
        
        residual_connection_0 = input + attention_input_normalized
        
        second_input_normalized = self.layer_norm_1(residual_connection_0)
        ff_out = self.linear(second_input_normalized)
        
        residual_connection_1 = residual_connection_0 + ff_out
        
        return residual_connection_1