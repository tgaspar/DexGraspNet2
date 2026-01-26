"""
MinkowskiEngine-based backbone networks for DexGraspNet2.

This module provides sparse convolutional backbones for point cloud
feature extraction using MinkowskiEngine.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

try:
    import MinkowskiEngine as ME
    from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
    HAS_MINKOWSKI = True
except ImportError:
    HAS_MINKOWSKI = False
    ME = None
    BasicBlock = None
    Bottleneck = None

from dexgraspnet2.configs.model_config import BackboneConfig

logger = logging.getLogger(__name__)


class BackboneBase(ABC, nn.Module):
    """
    Abstract base class for point cloud backbones.

    All backbones should inherit from this class and implement
    the forward method.

    Attributes:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize the backbone.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output feature channels.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, x: "ME.SparseTensor") -> "ME.SparseTensor":
        """
        Forward pass through the backbone.

        Args:
            x: Input sparse tensor.

        Returns:
            Output sparse tensor with features.
        """
        pass


class ResNetBase(nn.Module):
    """
    Base class for MinkowskiEngine ResNet backbones.

    This implements a standard ResNet architecture adapted for
    sparse 3D convolutions.
    """

    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)

    def __init__(self, in_channels: int, out_channels: int, D: int = 3):
        """
        Initialize the ResNet backbone.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output feature channels.
            D: Number of spatial dimensions (default: 3).
        """
        if not HAS_MINKOWSKI:
            raise ImportError("MinkowskiEngine is required for ResNet backbone")

        super().__init__()
        self.D = D
        assert self.BLOCK is not None, "BLOCK must be defined in subclass"

        self._network_initialization(in_channels, out_channels, D)
        self._weight_initialization()

    def _network_initialization(self, in_channels: int, out_channels: int, D: int):
        """Initialize network layers."""
        self.inplanes = self.INIT_DIM
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiInstanceNorm(self.inplanes),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=D),
        )

        self.layer1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=2)
        self.layer2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2)
        self.layer3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2], stride=2)
        self.layer4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3], stride=2)

        self.conv5 = nn.Sequential(
            ME.MinkowskiDropout(),
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=3, dimension=D
            ),
            ME.MinkowskiInstanceNorm(self.inplanes),
            ME.MinkowskiGELU(),
        )

        self.glob_pool = ME.MinkowskiGlobalMaxPooling()
        self.final = ME.MinkowskiLinear(self.inplanes, out_channels, bias=True)

    def _weight_initialization(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilation: int = 1,
        bn_momentum: float = 0.1,
    ) -> nn.Sequential:
        """Create a residual layer."""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D,
                ),
                ME.MinkowskiBatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dimension=self.D,
            )
        )
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, stride=1, dilation=dilation, dimension=self.D)
            )

        return nn.Sequential(*layers)

    def forward(self, x: "ME.SparseTensor") -> "ME.SparseTensor":
        """Forward pass through ResNet."""
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        x = self.glob_pool(x)
        return self.final(x)


class ResNet14D(ResNetBase):
    """ResNet-14 variant with custom plane dimensions for global features."""

    BLOCK = BasicBlock if HAS_MINKOWSKI else None
    LAYERS = (1, 1, 1, 1)
    PLANES = (32, 64, 128, 256, 192, 192, 192, 192)


class MinkUNetBase(ResNetBase):
    """
    Base class for MinkowskiEngine UNet backbones.

    This implements a UNet architecture with skip connections
    for dense point-wise feature prediction.
    """

    BLOCK = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    def __init__(self, in_channels: int, out_channels: int, D: int = 3):
        """
        Initialize the MinkUNet backbone.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output feature channels.
            D: Number of spatial dimensions (default: 3).
        """
        if not HAS_MINKOWSKI:
            raise ImportError("MinkowskiEngine is required for MinkUNet backbone")

        # Skip ResNetBase init, go directly to nn.Module
        nn.Module.__init__(self)
        self.D = D
        assert self.BLOCK is not None, "BLOCK must be defined in subclass"

        self._network_initialization(in_channels, out_channels, D)
        self._weight_initialization()

    def _network_initialization(self, in_channels: int, out_channels: int, D: int):
        """Initialize UNet network layers."""
        self.inplanes = self.INIT_DIM

        # Encoder
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D
        )
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D
        )
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D
        )
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D
        )
        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D
        )
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3])

        # Decoder
        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D
        )
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4], self.LAYERS[4])

        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D
        )
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5], self.LAYERS[5])

        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D
        )
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6], self.LAYERS[6])

        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D
        )
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7], self.LAYERS[7])

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=D,
        )
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: "ME.SparseTensor") -> "ME.SparseTensor":
        """
        Forward pass through MinkUNet.

        Args:
            x: Input sparse tensor.

        Returns:
            Output sparse tensor with point-wise features.
        """
        # Encoder path
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # Decoder path with skip connections
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)
        out = ME.cat(out, out_b3p8)
        out = self.block5(out)

        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)
        out = ME.cat(out, out_b2p4)
        out = self.block6(out)

        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)
        out = ME.cat(out, out_b1p2)
        out = self.block7(out)

        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)
        out = ME.cat(out, out_p1)
        out = self.block8(out)

        return self.final(out)


class MinkUNet14(MinkUNetBase):
    """MinkUNet-14 with basic blocks."""

    BLOCK = BasicBlock if HAS_MINKOWSKI else None
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class MinkUNet14D(MinkUNet14):
    """MinkUNet-14D variant with 192-channel decoder planes."""

    PLANES = (32, 64, 128, 256, 192, 192, 192, 192)


def get_backbone(
    backbone_name: str,
    feature_dim: int,
    backbone_config: Optional[Dict] = None,
) -> nn.Module:
    """
    Create a backbone network by name.

    Args:
        backbone_name: Name of the backbone ('sparseconv', 'sparse_glob_conv').
        feature_dim: Output feature dimension.
        backbone_config: Optional backbone-specific configuration.

    Returns:
        Initialized backbone network.

    Raises:
        ValueError: If backbone_name is not supported.
    """
    if not HAS_MINKOWSKI:
        raise ImportError(
            "MinkowskiEngine is required for backbone networks. "
            "Install with: pip install MinkowskiEngine"
        )

    if backbone_name == "sparseconv":
        return MinkUNet14D(in_channels=3, out_channels=feature_dim, D=3)
    elif backbone_name == "sparse_glob_conv":
        return ResNet14D(in_channels=3, out_channels=feature_dim, D=3)
    else:
        raise ValueError(
            f"Backbone '{backbone_name}' not supported. "
            f"Valid options: 'sparseconv', 'sparse_glob_conv'"
        )


def get_feature(
    backbone_name: str,
    backbone: nn.Module,
    data: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Extract features from point cloud using backbone.

    Args:
        backbone_name: Name of the backbone architecture.
        backbone: Initialized backbone network.
        data: Dictionary containing:
            - point_clouds: (B, N, 3) point coordinates
            - coors: (M, 4) sparse coordinates with batch index
            - feats: (M, C) sparse features
            - quantize2original: Mapping from quantized to original points

    Returns:
        (B, N, C) point-wise features.

    Raises:
        ValueError: If backbone_name is not supported.
    """
    if not HAS_MINKOWSKI:
        raise ImportError("MinkowskiEngine is required for feature extraction")

    pc = data["point_clouds"]
    batch_size, point_num, _ = pc.shape

    if backbone_name in ["sparseconv", "sparse_glob_conv"]:
        coor = data["coors"]
        feat = data["feats"]
        mink_input = ME.SparseTensor(feat, coordinates=coor)
        mink_output = backbone(mink_input).F

        if backbone_name == "sparseconv":
            # Map back to original point indices
            feature = mink_output[data["quantize2original"]].view(batch_size, point_num, -1)
        else:
            # Global feature - already pooled
            feature = mink_output
    else:
        raise ValueError(f"Backbone '{backbone_name}' not supported")

    return feature


def create_backbone_from_config(config: BackboneConfig) -> nn.Module:
    """
    Create a backbone network from configuration.

    Args:
        config: Backbone configuration dataclass.

    Returns:
        Initialized backbone network.
    """
    return get_backbone(
        backbone_name=config.name,
        feature_dim=config.out_channels,
        backbone_config=None,
    )
