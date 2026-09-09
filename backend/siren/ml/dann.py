"""Domain-Adversarial Neural Network (DANN) for terrain-invariant features (V3 §2.7).

The OOD (out-of-distribution) collapse from 0.67 → 0.24 IoU occurs because
Sen1Floods11 training chips are predominantly flat terrain, while the
deployment domain is steep Himalayan topography. The feature extractor
learns flat-terrain-specific representations that do not transfer.

DANN adds a Gradient Reversal Layer (GRL) and a domain discriminator:
  1. Feature extractor: the WaterResUNet encoder (shared, primary task).
  2. Segmentation head: predicts the water mask (primary task).
  3. Domain discriminator: binary classifier predicting whether a chip is
     "flat benchmark" (Sen1Floods11) or "Himalayan" (deployment domain).
  4. Gradient reversal: the discriminator's gradients are reversed before
     reaching the feature extractor, so the encoder learns features that
     are invariant to the terrain domain.

The domain discriminator uses only the terrain label (flat vs Himalayan),
not water labels — so the Himalayan chips need no annotation (semi-supervised
domain adaptation, V3 §2.7).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Autograd function that reverses gradients during backprop.

    The forward pass is identity; the backward pass negates the gradient
    and scales by ``lambda_`` (the domain-adaptation hyperparameter).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.lambda_, None


def gradient_reversal(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Apply the Gradient Reversal Layer.

    Args:
        x: input tensor.
        lambda_: gradient reversal strength (0 = no reversal, 1 = full).
    """
    return GradientReversalFunction.apply(x, lambda_)


class DomainDiscriminator(nn.Module):
    """Binary domain classifier: flat benchmark (0) vs Himalayan (1).

    A lightweight MLP over pooled encoder features. The input is the
    bottleneck feature vector from the WaterResUNet encoder (or any
    shared feature extractor). The output is a single logit (binary
    classification via sigmoid).

    Args:
        in_features: dimension of the input feature vector (encoder
            bottleneck channels, e.g. 512 for base_channels=32).
        hidden_dim: hidden layer dimension.
    """

    def __init__(self, in_features: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Classify pooled features into domain logits.

        Args:
            features: (B, in_features) pooled encoder features.

        Returns:
            (B, 1) domain logits (sigmoid → flat/Himalayan probability).
        """
        return self.net(features)


class DANNWrapper(nn.Module):
    """Wraps a segmentation model + domain discriminator for DANN training.

    The segmentation model's encoder produces shared features. These
    features feed both the segmentation head (primary task) and, via the
    gradient reversal layer, the domain discriminator (auxiliary task).

    During training:
        loss = loss_segmentation + lambda_domain * loss_domain

    The gradient reversal ensures the encoder learns domain-invariant
    features (the discriminator's gradients are reversed before reaching
    the encoder, so the encoder is trained to *fool* the discriminator).

    Args:
        segmenter: the segmentation model (e.g. WaterResUNet). Must expose
            an ``encode(x)`` method returning bottleneck features and a
            ``decode(bottleneck, skips)`` method, or a ``forward(x)``
            returning logits + a ``encoder_features(x)`` returning the
            pooled bottleneck.
        in_features: dimension of the bottleneck features for the domain
            discriminator.
        lambda_domain: gradient reversal strength (schedule this during
            training: start at 0, ramp to 1).
    """

    def __init__(
        self,
        segmenter: nn.Module,
        in_features: int,
        lambda_domain: float = 1.0,
    ) -> None:
        super().__init__()
        self.segmenter = segmenter
        self.discriminator = DomainDiscriminator(in_features)
        self.lambda_domain = lambda_domain

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (segmentation_logits, domain_logits).

        Args:
            x: (B, C, H, W) input tensor.

        Returns:
            (seg_logits, domain_logits) where seg_logits is (B, 1, H, W)
            and domain_logits is (B, 1).
        """
        # Get segmentation logits and bottleneck features from the segmenter
        seg_logits, bottleneck = self._forward_with_features(x)

        # Pool bottleneck for the domain discriminator
        pooled = F.adaptive_avg_pool2d(bottleneck, 1).flatten(1)  # (B, C)
        reversed_features = gradient_reversal(pooled, self.lambda_domain)
        domain_logits = self.discriminator(reversed_features)

        return seg_logits, domain_logits

    def _forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the segmenter and return both logits and bottleneck features.

        This assumes the segmenter exposes ``forward_with_features(x)``
        returning (logits, bottleneck). If not, falls back to forward(x)
        and re-extracts features (less efficient).
        """
        if hasattr(self.segmenter, "forward_with_features"):
            return self.segmenter.forward_with_features(x)
        # Fallback: forward only (domain head gets zeros — not useful)
        logits = self.segmenter(x)
        return logits, torch.zeros(
            x.shape[0], 1, device=x.device, requires_grad=False
        )


def domain_loss(domain_logits: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy loss for domain classification.

    Args:
        domain_logits: (B, 1) raw logits from the discriminator.
        domain_labels: (B, 1) binary labels {0=flat, 1=Himalayan}.

    Returns:
        Scalar BCE loss.
    """
    return F.binary_cross_entropy_with_logits(domain_logits, domain_labels.float())
