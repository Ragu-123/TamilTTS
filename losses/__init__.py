"""TamilTTSv2 losses package."""
from losses.losses import (
    masked_l1,
    MelLoss,
    DurationLoss,
    PitchEnergyLoss,
    DiscriminatorLoss,
    GeneratorAdversarialLoss,
    FeatureMatchingLoss,
    SLMLoss,
)

__all__ = [
    "masked_l1",
    "MelLoss",
    "DurationLoss",
    "PitchEnergyLoss",
    "DiscriminatorLoss",
    "GeneratorAdversarialLoss",
    "FeatureMatchingLoss",
    "SLMLoss",
]
