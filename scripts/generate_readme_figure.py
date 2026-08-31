#!/usr/bin/env python3
"""
Regenerate the README hero figure (assets/example.png).

Carves three 1,000-point subsets (uniform, gaussian, triangular targets)
out of the bundled 6-dimensional dataset and arranges the four scatterplot
matrices (original + three subsets) in a 2x2 grid.

Run from the repo root:
    MPLBACKEND=Agg .venv/bin/python scripts/generate_readme_figure.py
"""

import tempfile
from pathlib import Path

import scipy.io
from PIL import Image

from datacarve import plot_scatter_matrix, undersample_dataset

REPO = Path(__file__).parent.parent
DATA_FILE = REPO / "examples" / "data" / "DATA_random_6D.mat"
OUTPUT = REPO / "assets" / "example.png"

TARGETS = ["uniform", "gaussian", "triangular"]


def main() -> None:
    data = scipy.io.loadmat(DATA_FILE)["A"]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        panels = []

        # panel 1: the original dataset
        path = tmp / "original.png"
        plot_scatter_matrix(
            data,
            title=f"Original dataset ({data.shape[0]:,} datapoints)",
            save_path=str(path),
        )
        panels.append(path)

        # panels 2-4: carved subsets with different targets
        for target in TARGETS:
            print(f"Carving {target} subset...")
            mask = undersample_dataset(
                data,
                data_to_keep=1000,
                target_distribution=target,
                solver="SAT",  # best quality on this hard instance
                max_solver_time_sec=60,
                verbose=False,
                scatterplot_matrix=False,
            )
            path = tmp / f"{target}.png"
            plot_scatter_matrix(
                data[mask],
                title=f"Carved subset (1,000 datapoints) - {target}",
                save_path=str(path),
            )
            panels.append(path)

        # arrange the four panels in a 2x2 grid
        images = [Image.open(p) for p in panels]
        w = min(im.width for im in images)
        h = min(im.height for im in images)
        images = [im.resize((w, h)) for im in images]

        grid = Image.new("RGB", (2 * w, 2 * h), "white")
        for i, im in enumerate(images):
            grid.paste(im, ((i % 2) * w, (i // 2) * h))

        # halve the resolution to keep the README light (~1 MB)
        grid = grid.resize((w, h), Image.LANCZOS)
        grid.save(OUTPUT, optimize=True)

    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
