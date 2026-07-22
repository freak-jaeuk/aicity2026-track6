"""Cross-city Domain-Generalization (DG) augmentation presets for AI City Track 6.

Track 6 trains on one city and is scored on a HIDDEN, UNSEEN city. The dominant
domain shift is appearance/illumination/camera-ISP (color, white balance, exposure,
sensor noise, compression) plus moderate viewpoint differences across 36 cameras.

External data is forbidden and the target city is unavailable, so FDA / style-transfer
that needs target-domain reference images is NOT applicable. The practical substitute
is strong *photometric domain randomization* (+ mild geometry, + sensor-noise sim) so
the model stops keying on the source city's specific colour/lighting statistics.

These dicts are RF-DETR ``aug_config`` values. All keys used here are supported by both
the Albumentations (CPU) and Kornia (GPU) backends, so ``augmentation_backend="auto"``
can offload them to the (idle) GPU and relieve the data-loading-bound CPU pipeline.

Geometry is kept mild for upright fixed traffic cameras: NO VerticalFlip, small rotation
only (cars/people are not seen upside-down or strongly rotated).
"""

# RF-DETR stock default = HorizontalFlip only (≈ no domain randomization).
AUG_DEFAULT = {"HorizontalFlip": {"p": 0.5}}

# Recommended cross-city DG: strong photometric + sensor noise + mild geometry.
AUG_DG_CROSSCITY = {
    "HorizontalFlip": {"p": 0.5},
    "ColorJitter": {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.08, "p": 0.8},
    "RandomBrightnessContrast": {"brightness_limit": 0.3, "contrast_limit": 0.3, "p": 0.5},
    "GaussNoise": {"std_range": (0.01, 0.05), "p": 0.3},
    "GaussianBlur": {"blur_limit": 3, "p": 0.2},
    "Affine": {"scale": (0.85, 1.2), "translate_percent": (-0.06, 0.06), "rotate": (-7, 7), "p": 0.4},
}

# More aggressive photometric/noise — experimental (try if dg_crosscity helps).
AUG_DG_STRONG = {
    "HorizontalFlip": {"p": 0.5},
    "ColorJitter": {"brightness": 0.5, "contrast": 0.5, "saturation": 0.5, "hue": 0.12, "p": 0.9},
    "RandomBrightnessContrast": {"brightness_limit": 0.4, "contrast_limit": 0.4, "p": 0.6},
    "GaussNoise": {"std_range": (0.01, 0.08), "p": 0.4},
    "GaussianBlur": {"blur_limit": 5, "p": 0.3},
    "Affine": {"scale": (0.8, 1.25), "translate_percent": (-0.08, 0.08), "rotate": (-10, 10), "p": 0.5},
}

# Tuned cross-city DG v2: richer *photometric/ISP* randomization (gamma, channel
# shuffle = white-balance decorrelation, JPEG compression, ISO noise, CLAHE) with
# GENTLER geometry than dg_strong (rotate ±4, hue ±0.05) — dg_strong's hue 0.12
# (±43°) + rotate ±10 on fixed upright traffic cameras manufactures a distribution
# the unseen target city never contains. Validated to build+run with bboxes on the
# pinned albumentations 2.0.8. NOTE: CPU (Albumentations) backend only — several of
# these keys are not in the Kornia/GPU registry.
AUG_DG_CROSSCITY_V2 = {
    "HorizontalFlip": {"p": 0.5},
    "ColorJitter": {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.05, "p": 0.8},
    "RandomGamma": {"gamma_limit": (80, 120), "p": 0.3},
    "ChannelShuffle": {"p": 0.15},
    "ToGray": {"p": 0.02},
    "ImageCompression": {"quality_range": (40, 90), "p": 0.3},
    "ISONoise": {"p": 0.25},
    "GaussNoise": {"std_range": (0.01, 0.05), "p": 0.2},
    "GaussianBlur": {"blur_limit": 3, "p": 0.12},
    "CLAHE": {"clip_limit": 2.0, "p": 0.1},
    "Affine": {"scale": (0.9, 1.1), "translate_percent": (-0.05, 0.05), "rotate": (-4, 4), "p": 0.3},
}

AUG_PRESETS = {
    "default": AUG_DEFAULT,
    "dg_crosscity": AUG_DG_CROSSCITY,
    "dg_strong": AUG_DG_STRONG,
    "dg_crosscity_v2": AUG_DG_CROSSCITY_V2,
}


def resolve_aug_preset(name: str) -> dict:
    """Return the aug_config dict for a preset name, or raise on unknown name."""
    if name not in AUG_PRESETS:
        raise ValueError(f"Unknown aug_preset '{name}'. Options: {list(AUG_PRESETS)}")
    return AUG_PRESETS[name]
