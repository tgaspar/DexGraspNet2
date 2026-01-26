"""
MLP denoiser networks for diffusion models.

This module provides MLP-based denoising networks used in the
diffusion process for grasp pose generation.
"""

import logging
import math
from copy import deepcopy
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Mish(nn.Module):
    """
    Mish activation function.

    Mish(x) = x * tanh(softplus(x))

    This activation function has been shown to work well in
    diffusion models and provides smooth gradients.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Mish activation.

        Args:
            x: Input tensor.

        Returns:
            Activated tensor.
        """
        return x * torch.tanh(F.softplus(x))


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding for timesteps.

    This embedding encodes continuous timestep values into a
    high-dimensional representation using sine and cosine functions
    at different frequencies, similar to transformer positional encoding.

    Args:
        dim: Output embedding dimension.
        theta: Base for frequency computation (default: 10000).
    """

    def __init__(self, dim: int, theta: int = 10000):
        """Initialize the positional embedding."""
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute positional embedding.

        Args:
            x: (B,) tensor of timestep values in [0, 1].

        Returns:
            (B, dim) tensor of positional embeddings.
        """
        device = x.device
        half_dim = self.dim // 2

        # Compute frequency bands
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # Apply to input
        emb = x[:, None] * emb[None, :] * self.theta
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable architecture.

    Args:
        input_dim: Input feature dimension.
        hidden_layers_dim: List of hidden layer dimensions.
        output_dim: Output dimension.
        act: Activation function name ('relu', 'leaky_relu', 'mish', 'elu', 'tanh').
        use_layer_norm: Whether to apply layer normalization.

    Example:
        >>> mlp = MLP(input_dim=512, hidden_layers_dim=[256, 128], output_dim=64)
        >>> out = mlp(torch.randn(32, 512))
        >>> out.shape
        torch.Size([32, 64])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers_dim: List[int],
        output_dim: int,
        act: Optional[str] = None,
        use_layer_norm: bool = False,
    ):
        """Initialize the MLP."""
        super().__init__()

        # Select activation function
        act = act or "leaky_relu"
        act_fn = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "mish": Mish,
            "elu": nn.ELU,
            "tanh": nn.Tanh,
        }[act]

        # Build network
        hidden_layers_dim = deepcopy(hidden_layers_dim)
        hidden_layers_dim.insert(0, input_dim)

        self.mlp = nn.Sequential()
        for i in range(1, len(hidden_layers_dim)):
            self.mlp.add_module(
                f"linear{i - 1}",
                nn.Linear(hidden_layers_dim[i - 1], hidden_layers_dim[i]),
            )
            if use_layer_norm:
                self.mlp.add_module(f"ln{i - 1}", nn.LayerNorm(hidden_layers_dim[i]))
            self.mlp.add_module(f"act{i - 1}", act_fn())

        self.mlp.add_module(
            f"linear{len(hidden_layers_dim) - 1}",
            nn.Linear(hidden_layers_dim[-1], output_dim),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x: (B, input_dim) input tensor.

        Returns:
            (B, output_dim) output tensor.
        """
        return self.mlp(x)


class MLPDenoiser(MLP):
    """
    MLP-based denoiser for diffusion models.

    This network predicts the noise (or velocity) added to samples
    at each diffusion timestep, conditioned on point features.

    The network receives:
    - x: Current noisy sample
    - t: Timestep (normalized to [0, 1])
    - cond: Conditioning features from backbone

    Args:
        channels: Dimension of the sample being denoised.
        feature_dim: Dimension of conditioning features.
        hidden_layers_dim: Hidden layer dimensions for MLP.
        output_dim: Output dimension (should match channels).
        act: Activation function name.

    Example:
        >>> denoiser = MLPDenoiser(
        ...     channels=28,  # 9 (rot) + 3 (trans) + 16 (joints)
        ...     feature_dim=512,
        ...     hidden_layers_dim=[512, 256],
        ...     output_dim=28
        ... )
        >>> x = torch.randn(32, 28)  # Noisy samples
        >>> t = torch.rand(32)  # Timesteps
        >>> cond = torch.randn(32, 512)  # Features
        >>> pred = denoiser(x, t, cond)
        >>> pred.shape
        torch.Size([32, 28])
    """

    def __init__(
        self,
        channels: int,
        feature_dim: int,
        hidden_layers_dim: List[int],
        output_dim: int,
        act: str = "mish",
    ):
        """Initialize the denoiser."""
        self.channels = channels
        input_dim = channels + feature_dim

        super().__init__(
            input_dim=input_dim,
            hidden_layers_dim=hidden_layers_dim,
            output_dim=output_dim,
            act=act,
        )

        # Timestep embedding
        self.embedding = SinusoidalPosEmb(feature_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise/velocity for denoising.

        Args:
            x: (B, channels) noisy sample.
            t: (B,) normalized timesteps in [0, 1].
            cond: (B, feature_dim) conditioning features.

        Returns:
            (B, output_dim) predicted noise or velocity.
        """
        # Embed timestep and add to conditioning
        t_emb = self.embedding(t)
        cond_with_time = cond + t_emb

        # Concatenate sample with conditioned features
        mlp_input = torch.cat([x, cond_with_time], dim=-1)

        return super().forward(mlp_input)


# Alias for backward compatibility
MLPWrapper = MLPDenoiser
