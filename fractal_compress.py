#!/usr/bin/env python3
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_PATH = Path("Lenna.png")
OUTPUT_PATH = Path("indices.npy")
IMAGE_SIZE = 512
RANGE_SIZE = 4
DOMAIN_SIZE = 8
SYMMETRY_COUNT = 8


def extract_blocks(image: np.ndarray, size: int) -> np.ndarray:
    blocks = []
    for y in range(0, IMAGE_SIZE, size):
        for x in range(0, IMAGE_SIZE, size):
            blocks.append(image[y : y + size, x : x + size])
    return np.stack(blocks, axis=0)


def transform(arr: np.ndarray, sym: int) -> np.ndarray:
    rotated = np.rot90(arr, k=sym % 4, axes=(-2, -1))
    if sym >= 4:
        rotated = np.flip(rotated, axis=-1)
    return rotated


def downscale(arr: np.ndarray) -> np.ndarray:
    return (
        arr[..., ::2, ::2]
        + arr[..., ::2, 1::2]
        + arr[..., 1::2, ::2]
        + arr[..., 1::2, 1::2]
    ) * 0.25


def main() -> None:
    image = np.asarray(Image.open(IMAGE_PATH).convert("L"), dtype=np.float64)

    domain_blocks = extract_blocks(image, DOMAIN_SIZE).astype(np.float64)
    domain_variants = np.stack(
        [downscale(transform(domain_blocks, sym)) for sym in range(SYMMETRY_COUNT)],
        axis=1,
    )
    variant_flat = domain_variants.reshape(-1, RANGE_SIZE * RANGE_SIZE)
    variant_mean = variant_flat.mean(axis=1)
    variant_centered = variant_flat - variant_mean[:, None]
    variant_energy = np.sum(variant_centered * variant_centered, axis=1)

    range_blocks_per_side = IMAGE_SIZE // RANGE_SIZE
    pairs = np.empty(
        range_blocks_per_side * range_blocks_per_side,
        dtype=[("domain", "i4"), ("sym", "i4"), ("contrast", "f4"), ("offset", "f4")],
    )

    for idx, y in enumerate(range(0, IMAGE_SIZE, RANGE_SIZE)):
        base = idx * range_blocks_per_side
        for jdx, x in enumerate(range(0, IMAGE_SIZE, RANGE_SIZE)):
            block = image[y : y + RANGE_SIZE, x : x + RANGE_SIZE]
            target = block.reshape(-1)
            target_mean = target.mean()
            target_centered = target - target_mean
            target_energy = np.sum(target_centered * target_centered)

            numerator = np.einsum("ij,j->i", variant_centered, target_centered)

            with np.errstate(divide="ignore", invalid="ignore"):
                contrast = numerator / variant_energy
            contrast = np.where(variant_energy > 1e-6, contrast, 0.0)

            errors = np.where(
                variant_energy > 1e-6,
                target_energy - (numerator * numerator) / variant_energy,
                target_energy,
            )
            errors = np.maximum(errors, 0.0)

            best = int(np.argmin(errors))
            mean_d = variant_mean[best]
            best_contrast = float(contrast[best])
            best_offset = float(target_mean - best_contrast * mean_d)

            pairs[base + jdx] = (
                best // SYMMETRY_COUNT,
                best % SYMMETRY_COUNT,
                best_contrast,
                best_offset,
            )

    np.save(OUTPUT_PATH, pairs)
    

if __name__ == "__main__":
    main()
