from collections.abc import Callable
from pathlib import Path

import pandas as pd

from drscreen.data.inspection import class_distribution, render_sample_grid
from drscreen.data.schema import GRADE, IMAGE_PATH, SPLIT

NAMES = ("zero", "one", "two")


def test_class_distribution_counts_every_class_and_split() -> None:
    manifest = pd.DataFrame({GRADE: [0, 0, 1, 0], SPLIT: ["train", "val", "train", "train"]})

    table = class_distribution(manifest, GRADE, NAMES)

    assert table.index.tolist() == list(NAMES)
    assert table.loc["zero", "train"] == 2
    assert table.loc["zero", "val"] == 1
    assert table.loc["two", "total"] == 0
    assert table["percent"].sum() == 100.0


def test_sample_grid_has_one_row_per_class(
    tmp_path: Path, write_fundus: Callable[[Path], Path]
) -> None:
    paths = [str(write_fundus(tmp_path / f"{i}.png")) for i in range(3)]
    manifest = pd.DataFrame({IMAGE_PATH: paths, GRADE: [0, 0, 1]})

    figure = render_sample_grid(
        manifest, GRADE, NAMES, samples_per_class=2, thumbnail_size=16, seed=0
    )

    assert len(figure.axes) == len(NAMES) * 2
    shown = [axis for axis in figure.axes if axis.images]
    assert len(shown) == 3  # two grade-0 images, one grade-1, none for grade 2
    figure.savefig(tmp_path / "grid.png")
    assert (tmp_path / "grid.png").stat().st_size > 0
