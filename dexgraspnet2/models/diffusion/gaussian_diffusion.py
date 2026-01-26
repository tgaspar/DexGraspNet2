"""
Gaussian diffusion model for grasp pose generation.

This module implements DDPM (Denoising Diffusion Probabilistic Models)
with support for velocity prediction and log probability estimation.
"""

import logging
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
except ImportError:
    DDPMScheduler = None

from dexgraspnet2.configs.model_config import DiffusionConfig

logger = logging.getLogger(__name__)


def jacobian_matrix(f: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Compute the Jacobian matrix df/dz.

    Args:
        f: (B, D) output tensor.
        z: (B, D) input tensor with requires_grad=True.

    Returns:
        (B, D, D) Jacobian matrix.
    """
    jacobian = torch.zeros((*f.shape, z.shape[-1]), device=f.device)
    for i in range(f.shape[-1]):
        grad = torch.autograd.grad(
            f[..., i].sum(),
            z,
            retain_graph=(i != f.shape[-1] - 1),
            allow_unused=True,
        )[0]
        if grad is not None:
            jacobian[..., i, :] = grad
    return jacobian.contiguous()


def approx_jacobian_trace(f: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Approximate Jacobian trace using Hutchinson's estimator.

    This is much faster than computing the full Jacobian matrix,
    using random projections to estimate the trace.

    Args:
        f: (B, D) output tensor.
        z: (B, D) input tensor with requires_grad=True.

    Returns:
        (B,) estimated trace values.
    """
    e = torch.normal(mean=0, std=1, size=f.shape, device=f.device, dtype=f.dtype)
    grad = torch.autograd.grad(f, z, grad_outputs=e)[0]
    return torch.einsum("nd,nd->n", e, grad)


def jacobian_trace(
    log_prob_type: Optional[str],
    dx: torch.Tensor,
    dy: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Jacobian trace for log probability estimation.

    Args:
        log_prob_type: Type of estimation ('accurate_cont', 'estimate', or None).
        dx: Input perturbation tensor.
        dy: Output change tensor.

    Returns:
        Jacobian trace estimate, or 0 if log_prob_type is None.
    """
    if log_prob_type == "accurate_cont":
        # Accurate but slow - compute full Jacobian
        jacobian_mat = jacobian_matrix(dy, dx)
        return jacobian_mat.diagonal(dim1=-1, dim2=-2).sum(dim=-1)
    elif log_prob_type == "estimate":
        # Fast approximate trace using Hutchinson's estimator
        return approx_jacobian_trace(dy, dx)
    else:
        return 0


class GaussianDiffusion1D(nn.Module):
    """
    1D Gaussian diffusion model for grasp pose generation.

    This implements DDPM with configurable noise schedules and
    prediction types (epsilon or velocity).

    The diffusion process models:
        q(x_t | x_0) = N(x_t; sqrt(alpha_t) * x_0, (1 - alpha_t) * I)

    And learns to reverse this process to generate samples.

    Args:
        model: Denoising network that predicts noise/velocity.
        config: Diffusion configuration.
        cond_fn: Optional function to transform conditioning during sampling.

    Example:
        >>> from dexgraspnet2.models.diffusion import MLPDenoiser
        >>> denoiser = MLPDenoiser(channels=28, feature_dim=512, ...)
        >>> diffusion = GaussianDiffusion1D(denoiser, config)
        >>> # Training: compute loss
        >>> loss = diffusion(x_target, cond_features)
        >>> # Inference: sample
        >>> samples, log_prob = diffusion.sample(cond_features)
    """

    def __init__(
        self,
        model: nn.Module,
        config: DiffusionConfig,
        cond_fn: Optional[Callable] = None,
    ):
        """Initialize the diffusion model."""
        super().__init__()

        if DDPMScheduler is None:
            raise ImportError(
                "diffusers is required for GaussianDiffusion1D. "
                "Install with: pip install diffusers"
            )

        self.config = config
        self.model = model
        self.cond_fn = cond_fn or (lambda x, t, cond: cond)

        # Initialize scheduler
        scheduler_kwargs = config.to_scheduler_dict()
        if config.scheduler_type == "DDPMScheduler":
            self.scheduler = DDPMScheduler(**scheduler_kwargs)
        else:
            raise NotImplementedError(f"Scheduler {config.scheduler_type} not supported")

        self.timesteps = config.num_train_timesteps
        self.inference_timesteps = config.num_inference_timesteps
        self.prediction_type = config.prediction_type

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Compute diffusion training loss.

        Args:
            x: (B, D) target samples.
            cond: (B, C) conditioning features.

        Returns:
            Scalar loss tensor.
        """
        return self.calculate_loss(x, cond)

    def calculate_loss(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Compute mean squared error loss for denoising.

        The loss is computed as:
            L = E_t,eps [ ||model(x_t, t, cond) - target||^2 ]

        where target is either epsilon (noise) or v (velocity).

        Args:
            x: (B, D) clean samples.
            cond: (B, C) conditioning features.

        Returns:
            Scalar MSE loss.
        """
        batch_size = x.shape[0]
        device = x.device

        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)

        # Sample noise and create noisy samples
        noise = torch.randn_like(x)
        noised_x = self.scheduler.add_noise(x, noise, t)

        # Get conditioning (potentially modified by cond_fn)
        t_normalized = t.float() / self.timesteps
        cond = self.cond_fn(noised_x, t_normalized, cond)

        # Predict
        pred = self.model(noised_x, t_normalized, cond=cond)

        # Compute target based on prediction type
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(x, noise, t)
        else:
            raise NotImplementedError(f"Prediction type {self.prediction_type} not supported")

        # MSE loss
        loss = (pred - target).square().mean()

        return loss

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate samples via reverse diffusion.

        Args:
            cond: (B, C) conditioning features.
            return_trajectory: Whether to return full denoising trajectory.

        Returns:
            Tuple of:
                - (B, D) generated samples
                - (B,) log probability estimates (if configured)
        """
        batch_size = cond.shape[0]
        device = cond.device

        # Start from pure noise
        x = torch.randn(batch_size, self.model.channels, device=device)

        # Initialize log probability (Gaussian prior)
        log_prob = (-x.square() / 2 - np.log(2 * np.pi) / 2).sum(1)

        # Set inference timesteps
        self.scheduler.set_timesteps(self.inference_timesteps, device=device)

        need_log_prob = self.config.log_prob_type is not None
        last_t = self.timesteps
        trajectory = [x.clone()] if return_trajectory else None

        with torch.set_grad_enabled(need_log_prob):
            for t in self.scheduler.timesteps:
                # Create perturbation for Jacobian computation
                dx = torch.zeros_like(x)
                dx.requires_grad_(need_log_prob)
                x = x + dx

                # Compute timestep info
                dt = torch.full(
                    (batch_size, 1),
                    (last_t - t.item()) / self.timesteps,
                    device=device,
                    dtype=torch.float,
                )
                last_t = t.item()

                t_batch = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)
                t_normalized = t_batch.float() / self.timesteps

                # Get model prediction
                cond_now = self.cond_fn(x, t_normalized, cond)
                model_output = self.model(x, t_normalized, cond=cond_now)

                # Get noise schedule parameters
                alpha_prod = self.scheduler.alphas_cumprod.to(device)[t_batch][:, None]
                betas = self.scheduler.betas.to(device)[t_batch][:, None]

                # Convert prediction to noise
                if self.prediction_type == "epsilon":
                    noise = model_output
                elif self.prediction_type == "v_prediction":
                    noise = model_output * alpha_prod.sqrt() + x * (1 - alpha_prod).sqrt()

                # Compute score function
                score = -1 / (1 - alpha_prod).sqrt() * noise
                beta = betas * self.timesteps

                # Reverse step (ODE or SDE)
                if self.config.ode:
                    # Deterministic ODE
                    dy = (-0.5 * beta * x - score * beta / 2) * dt
                else:
                    # Stochastic SDE
                    dy = (
                        (-0.5 * beta * x - score * beta) * dt
                        + beta.sqrt() * torch.randn_like(x) * dt.sqrt()
                    )

                # Update log probability
                log_prob -= jacobian_trace(self.config.log_prob_type, dx, -dy / dt) * dt[:, 0]

                # Update sample
                x = x - dy
                x = x.detach()
                log_prob = log_prob.detach()

                if return_trajectory:
                    trajectory.append(x.clone())

        if not need_log_prob:
            log_prob = log_prob * 0

        if return_trajectory:
            return x, log_prob, torch.stack(trajectory, dim=1)

        return x, log_prob

    def log_prob(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of samples (not implemented).

        Args:
            x: (B, D) samples.
            cond: (B, C) conditioning features.

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        raise NotImplementedError("Log probability computation not implemented")


def create_diffusion_model(
    config: DiffusionConfig,
    channels: int,
    feature_dim: int,
    hidden_dims: Optional[list] = None,
    activation: str = "mish",
) -> GaussianDiffusion1D:
    """
    Create a diffusion model with denoiser network.

    Args:
        config: Diffusion configuration.
        channels: Dimension of samples to generate.
        feature_dim: Dimension of conditioning features.
        hidden_dims: Hidden layer dimensions for denoiser MLP.
        activation: Activation function for MLP.

    Returns:
        Initialized GaussianDiffusion1D model.
    """
    from dexgraspnet2.models.diffusion.mlp_denoiser import MLPDenoiser

    hidden_dims = hidden_dims or [512, 256]

    denoiser = MLPDenoiser(
        channels=channels,
        feature_dim=feature_dim,
        hidden_layers_dim=hidden_dims,
        output_dim=channels,
        act=activation,
    )

    return GaussianDiffusion1D(model=denoiser, config=config)
