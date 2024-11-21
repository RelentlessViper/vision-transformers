# Import necessary dependencies
import einops
import torch
from torch import nn
from const import *

class PatchEmbedding(nn.Module):
    """
    Create Patch Embeddings from images.
    
    Attributes
    ----------
    `patch_size`: int = 16
        The size of patches.
    `n_channels`: int = 3
        The amount of channels in the image.
    `n_model`: int = 768
        The size of embeddings that is calculated in the following way: P * P * C.
    `batch_size`: int = 4
        The batch size.
    `device`: str = 'cpu'
        The device. Can be either 'cpu', or 'cuda:n', where 'n' is the number of GPU processors.
    """
    
    def __init__(
        self,
        patch_size: int = PATCH_SIZE,
        n_channels: int = N_CHANNELS,
        n_model: int = LATENT_SIZE,
        batch_size: int = BATCH_SIZE,
        device: str = 'cpu'
    ) -> None:
        """
        Initialize the PatchEmbedding class.
        
        Parameters
        ----------
        `patch_size`: int = 16
            The size of patches.
        `n_channels`: int = 3
            The amount of channels in the image.
        `n_model`: int = 768
            The size of embeddings that is calculated in the following way: P * P * C.
        `batch_size`: int = 4
            The batch size.
        `device`: str = 'cpu'
            The device. Can be either 'cpu', or 'cuda:n', where 'n' is the number of GPU processors.
        
        Returns
        ----------
        `self`: PatchEmbedding
            PatchEmbedding class object
        """
        
        super(PatchEmbedding, self).__init__()
        
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.n_model = n_model
        self.batch_size = batch_size
        self.device = device
        
        self.input_size = self.patch_size * self.patch_size * self.n_channels
        
        self.linear_projection = nn.Linear(self.input_size, self.n_model)
        
        # Class labels (random)
        self.class_label = nn.Parameter(torch.randn(self.batch_size, 1, self.n_model)).to(device)
        
        self.positional_embedding = nn.Parameter(torch.randn(self.batch_size, 1, self.n_model)).to(device)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Generate embeddings in the following way: \n
        `Input shape`: (b, c, h, w) \n
        `Output shape`: (b, n, n_model).

        The method takes an image with shape (b, c, h, w), where:
            `b` - batch_size;
            `c` - num_channels;
            `h` - height;
            `w` - width.
        
        The following transformations will occur:
            1) Transform the batch of images to the following form: (b, (h * w) / (patch_size ** 2), patch_size ** 2 * c);
            2) Use linear layer to create projections and change the shape to (b, n, n_model);
            3) Prepend the class embedding to each image, changing the shape to (b, n + 1, n_model);
            4) Generate positional embeddings with the same shape and add them to our images.
        
        Parameters
        ----------
        input: torch.Tensor()
            Input batch of images.
        
        Returns
        ----------
        linear_projection: torch.Tensor()
            Patch embeddings for all given images.
        """
        
        input = input.to(self.device)
        
        patches = einops.rearrange(
            input,
            'b c (h h1) (w w1) -> b (h w) (h1 w1 c)',
            h1=self.patch_size,
            w1=self.patch_size
        )
        
        linear_projection = self.linear_projection(patches).to(self.device)
        b, n, _ = linear_projection.shape
        
        linear_projection = torch.cat((self.class_label, linear_projection), dim=1)
        positional_embedding = einops.repeat(
            self.positional_embedding,
            "b 1 d -> b m d",
            m=n+1
        )
        
        linear_projection = linear_projection + positional_embedding
        
        return linear_projection