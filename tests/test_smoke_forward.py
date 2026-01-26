"""
Smoke test for DexGraspNet2 model forward/backward pass.

Verifies that the model can:
1. Be instantiated from config
2. Process a synthetic batch through forward pass
3. Compute gradients through backward pass
"""

import logging
import sys

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_synthetic_batch(
    batch_size: int = 2,
    num_points: int = 1000,
    num_grasps: int = 8,
    voxel_size: float = 0.005,
    device: torch.device = None,
):
    """
    Create a synthetic batch matching the expected data format.

    Args:
        batch_size: Number of scenes in batch.
        num_points: Number of points per scene.
        num_grasps: Number of grasp annotations per scene.
        voxel_size: Voxel size for quantization.
        device: Target device.

    Returns:
        Dictionary with synthetic batch data.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate random point clouds in [0, 1] range
    point_clouds = torch.rand(batch_size, num_points, 3, device=device) * 0.5

    # Quantize points to voxels
    voxel_coords = torch.floor(point_clouds / voxel_size).long()

    # Build sparse coordinates (batch_idx, x, y, z)
    coors_list = []
    quantize2original_list = []
    offset = 0

    for b in range(batch_size):
        coords_b = voxel_coords[b]  # (N, 3)
        # Get unique voxels and mapping
        unique_coords, inverse = torch.unique(
            coords_b, dim=0, return_inverse=True
        )
        # Add batch index
        batch_idx = torch.full(
            (unique_coords.shape[0], 1), b, device=device, dtype=torch.long
        )
        coors_b = torch.cat([batch_idx, unique_coords], dim=1)
        coors_list.append(coors_b)

        # Build quantize2original mapping (maps each original point to its sparse idx)
        quantize2original_list.append(inverse + offset)
        offset += unique_coords.shape[0]

    coors = torch.cat(coors_list, dim=0)
    quantize2original = torch.cat(quantize2original_list, dim=0)

    # Sparse features (XYZ coordinates normalized to unit sphere)
    # The backbone expects 3-channel input (in_channels=3)
    feats_list = []
    for b in range(batch_size):
        coords_b = voxel_coords[b]
        unique_coords, _ = torch.unique(coords_b, dim=0, return_inverse=True)
        # Use normalized voxel centers as features
        voxel_centers = (unique_coords.float() + 0.5) * voxel_size
        feats_list.append(voxel_centers)
    feats = torch.cat(feats_list, dim=0).to(device)

    # Ground truth labels
    objectness = torch.randint(0, 2, (batch_size, num_points), device=device).long()
    graspness = torch.rand(batch_size, num_points, device=device)

    # Grasp annotations
    centers = torch.randint(0, num_points, (batch_size, num_grasps), device=device)
    trans = point_clouds[
        torch.arange(batch_size, device=device)[:, None],
        centers
    ] + torch.randn(batch_size, num_grasps, 3, device=device) * 0.01

    # Random rotation matrices (identity + small perturbation -> orthogonalize)
    rot_base = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)
    rot_base = rot_base.expand(batch_size, num_grasps, 3, 3).clone()
    rot_noise = torch.randn(batch_size, num_grasps, 3, 3, device=device) * 0.1
    rot = rot_base + rot_noise
    # Orthogonalize via SVD
    u, _, vh = torch.linalg.svd(rot)
    rot = u @ vh

    # Joint angles
    qpos = torch.randn(batch_size, num_grasps, 16, device=device) * 0.5

    # Graspness availability
    has_graspness = torch.ones(batch_size, 1, device=device)

    return {
        "point_clouds": point_clouds,
        "coors": coors,
        "feats": feats,
        "quantize2original": quantize2original,
        "objectness": objectness,
        "graspness": graspness,
        "trans": trans,
        "rot": rot,
        "qpos": qpos,
        "centers": centers,
        "has_graspness": has_graspness,
    }


def test_forward_backward():
    """Test model forward and backward pass with synthetic data."""
    try:
        from dexgraspnet2.models import create_model
        from dexgraspnet2.configs import ModelConfig
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        logger.error("Make sure MinkowskiEngine and dependencies are installed")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model
    config = ModelConfig()
    logger.info(f"Model config: type={config.type}, feature_dim={config.feature_dim}")

    model = create_model(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    # Create synthetic batch
    batch = create_synthetic_batch(
        batch_size=2,
        num_points=1000,
        num_grasps=8,
        voxel_size=config.voxel_size,
        device=device,
    )
    logger.info(
        f"Batch: {batch['point_clouds'].shape[0]} scenes, "
        f"{batch['point_clouds'].shape[1]} points, "
        f"{batch['trans'].shape[1]} grasps"
    )

    # Forward pass
    model.train()
    loss, result_dict = model(batch)
    logger.info(f"Loss: {loss.item():.4f}")

    # Log individual losses
    for key, value in result_dict.items():
        if key.startswith("loss_"):
            logger.info(f"  {key}: {value.mean().item():.4f}")

    # Check for NaN
    if torch.isnan(loss):
        logger.error("Loss is NaN!")
        return False

    # Backward pass
    loss.backward()

    # Check gradients
    grad_norm = 0.0
    num_grads = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
            num_grads += 1

    grad_norm = grad_norm ** 0.5
    logger.info(f"Gradient norm: {grad_norm:.4f} ({num_grads} tensors with gradients)")

    if grad_norm == 0:
        logger.error("No gradients computed!")
        return False

    if not torch.isfinite(torch.tensor(grad_norm)):
        logger.error("Gradient norm is not finite!")
        return False

    logger.info("Smoke test passed!")
    return True


def test_model_creation():
    """Test that model can be created with various configs."""
    try:
        from dexgraspnet2.models import create_model
        from dexgraspnet2.configs import ModelConfig
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return False

    configs = [
        ModelConfig(),  # Default
        ModelConfig(dist_joint=0),  # Separate joint MLP
        ModelConfig(trans_scale=25.0),  # Paper default
    ]

    for i, config in enumerate(configs):
        try:
            model = create_model(config)
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"Config {i}: {param_count:,} parameters")
        except Exception as e:
            logger.error(f"Config {i} failed: {e}")
            return False

    logger.info("Model creation tests passed!")
    return True


def main():
    """Run all smoke tests."""
    logger.info("=" * 60)
    logger.info("DexGraspNet2 Smoke Test")
    logger.info("=" * 60)

    results = []

    # Test 1: Model creation
    logger.info("\n[1/2] Testing model creation...")
    results.append(("Model creation", test_model_creation()))

    # Test 2: Forward/backward pass
    logger.info("\n[2/2] Testing forward/backward pass...")
    results.append(("Forward/backward", test_forward_backward()))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary:")
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {name}: {status}")
        if not passed:
            all_passed = False

    logger.info("=" * 60)

    if all_passed:
        logger.info("All smoke tests passed!")
        return 0
    else:
        logger.error("Some smoke tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
