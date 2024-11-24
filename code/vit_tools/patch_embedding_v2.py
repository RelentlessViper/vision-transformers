# Import necessary dependencies
import torch
from torch import nn
from const import *

class PatchEmbedding(nn.Module):
    """
    Turns a 2D input image into a 1D sequence learnable embedding vector.

    Attributes
    ----------
        n_channels: int = 3
            Number of color channels for the input images.
        patch_size: int = 16
            Size of patches to convert input image into.
        n_model: int = 768
            Size of embedding to turn image into.
    """

    def __init__(
        self,
        patch_size: int = PATCH_SIZE,
        n_channels: int = N_CHANNELS,
        n_model: int = LATENT_SIZE,
        batch_size: int = BATCH_SIZE
    ) -> None:
        """
        Initialize PatchEmbedding class.
        
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
        
        self.patcher = nn.Conv2d(in_channels=n_channels,
                                 out_channels=n_model,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)


        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, input) -> torch.Tensor:
        """
        Generate embeddings in the following way: \n
        `Input shape`: (b, c, h, w) \n
        `Output shape`: (b, n, n_model).
        
        Parameters
        ----------
        input: torch.Tensor()
            Input batch of images.
        
        Returns
        ----------
        linear_projection: torch.Tensor()
            Patch embeddings for all given images.
        """

        image_resolution = input.shape[-1]
        
        assert image_resolution % self.patch_size == 0, (
            f"Input image size must be divisible by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"
        )

        x_patched = self.patcher(input)
        x_flattened = self.flatten(x_patched)

        return x_flattened.permute(0, 2, 1)