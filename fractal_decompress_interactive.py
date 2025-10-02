#!/usr/bin/env python3
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

INDICES_PATH = Path("indices.npy")
IMAGE_SIZE = 512
RANGE_SIZE = 4
DOMAIN_SIZE = 8
SYMMETRY_COUNT = 8
SEED = 1234


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


def iterate(image: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    range_blocks_per_side = IMAGE_SIZE // RANGE_SIZE
    domain_blocks_per_side = IMAGE_SIZE // DOMAIN_SIZE

    new_image = np.empty_like(image, dtype=np.float64)

    domains = pairs["domain"]
    syms = pairs["sym"]
    contrasts = pairs["contrast"].astype(np.float64)
    offsets = pairs["offset"].astype(np.float64)

    for block_idx in range(pairs.shape[0]):
        ry = (block_idx // range_blocks_per_side) * RANGE_SIZE
        rx = (block_idx % range_blocks_per_side) * RANGE_SIZE

        domain_index = int(domains[block_idx])
        sym = int(syms[block_idx])
        dy = (domain_index // domain_blocks_per_side) * DOMAIN_SIZE
        dx = (domain_index % domain_blocks_per_side) * DOMAIN_SIZE

        domain_block = image[dy : dy + DOMAIN_SIZE, dx : dx + DOMAIN_SIZE]
        transformed = transform(domain_block, sym)
        downscaled = downscale(transformed)
        new_image[ry : ry + RANGE_SIZE, rx : rx + RANGE_SIZE] = (
            contrasts[block_idx] * downscaled + offsets[block_idx]
        )

    return new_image


def main() -> None:
    pairs = np.load(INDICES_PATH)
    rng = np.random.default_rng(SEED)
    image = rng.random((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float64) * 255.0

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    im = ax.imshow(np.clip(image, 0.0, 255.0), cmap="gray", vmin=0, vmax=255)
    ax.set_title("Iteration 0")
    ax.axis("off")

    iteration = {"count": 0, "image": image}

    def do_iteration(_=None) -> None:
        iteration["image"] = iterate(iteration["image"], pairs)
        iteration["count"] += 1
        im.set_data(np.clip(iteration["image"], 0.0, 255.0))
        ax.set_title(f"Iteration {iteration['count']}")
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        if event.key == " ":
            do_iteration()
        elif event.key == "s":
            result = np.clip(iteration["image"], 0.0, 255.0).astype(np.uint8)
            Image.fromarray(result).save("reconstructed_iterative.png")
            print("Saved reconstructed_iterative.png")

    button_ax = fig.add_axes([0.4, 0.05, 0.2, 0.075])
    button = Button(button_ax, "Iterate")
    button.on_clicked(do_iteration)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    main()
