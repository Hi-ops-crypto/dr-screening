from pathlib import Path

import pandas as pd
import pytest

from drscreen.data.schema import GRADE, GROUP, IMAGE_ID, IMAGE_PATH, QUALITY, SOURCE, Source
from drscreen.data.sources import parse_aptos, parse_ddr, parse_eyepacs, parse_eyeq, parse_idrid
from drscreen.exceptions import DataError


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_parse_aptos(tmp_path: Path) -> None:
    pd.DataFrame({"id_code": ["a1", "b2"], "diagnosis": [0, 4]}).to_csv(
        tmp_path / "train.csv", index=False
    )
    for name in ("a1", "b2"):
        _touch(tmp_path / "train_images" / f"{name}.png")

    manifest = parse_aptos(tmp_path)

    assert manifest[GRADE].tolist() == [0, 4]
    assert manifest[IMAGE_PATH].iloc[1] == str(tmp_path / "train_images" / "b2.png")
    assert set(manifest[SOURCE]) == {"aptos"}
    assert manifest[IMAGE_ID].tolist() == ["a1", "b2"]
    assert manifest[GROUP].tolist() == ["a1", "b2"]


def test_parse_aptos_missing_image_is_error(tmp_path: Path) -> None:
    pd.DataFrame({"id_code": ["a1"], "diagnosis": [2]}).to_csv(tmp_path / "train.csv", index=False)

    with pytest.raises(DataError, match="missing"):
        parse_aptos(tmp_path)


def test_parse_aptos_out_of_range_grade_is_error(tmp_path: Path) -> None:
    pd.DataFrame({"id_code": ["a1"], "diagnosis": [7]}).to_csv(tmp_path / "train.csv", index=False)
    _touch(tmp_path / "train_images" / "a1.png")

    with pytest.raises(DataError, match="outside"):
        parse_aptos(tmp_path)


def test_parse_aptos_missing_label_file_is_error(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="not found"):
        parse_aptos(tmp_path)


def _write_idrid(root: Path) -> None:
    """Mimic the nested mirror layout, including the official CSV quirks."""
    base = root / "B.%20Disease%20Grading" / "B. Disease Grading"
    truths = base / "2. Groundtruths"
    truths.mkdir(parents=True)
    # Official files carry a trailing-space column name and empty trailing columns.
    (truths / "a. IDRiD_Disease Grading_Training Labels.csv").write_text(
        "Image name,Retinopathy grade,Risk of macular edema ,,\n"
        "IDRiD_001,3,2,,\nIDRiD_002,0,0,,\n,,,,\n"
    )
    (truths / "b. IDRiD_Disease Grading_Testing Labels.csv").write_text(
        "Image name,Retinopathy grade,Risk of macular edema \nIDRiD_001,4,1\n"
    )
    _touch(base / "1. Original Images" / "a. Training Set" / "IDRiD_001.jpg")
    _touch(base / "1. Original Images" / "a. Training Set" / "IDRiD_002.jpg")
    _touch(base / "1. Original Images" / "b. Testing Set" / "IDRiD_001.jpg")
    # Other parts of the release reuse the same folder names and must be ignored.
    for part in ("A. Segmentation", "C. Localization"):
        for subset in ("a. Training Set", "b. Testing Set"):
            _touch(root / part / part / "1. Original Images" / subset / "IDRiD_001.jpg")


def test_parse_idrid_keeps_overlapping_subset_names_apart(tmp_path: Path) -> None:
    _write_idrid(tmp_path)

    manifest = parse_idrid(tmp_path).set_index(IMAGE_ID)

    assert sorted(manifest.index) == ["test_IDRiD_001", "train_IDRiD_001", "train_IDRiD_002"]
    assert manifest.loc["train_IDRiD_001", GRADE] == 3
    assert manifest.loc["test_IDRiD_001", GRADE] == 4
    assert "Testing Set" in str(manifest.loc["test_IDRiD_001", IMAGE_PATH])
    assert "Training Set" in str(manifest.loc["train_IDRiD_001", IMAGE_PATH])


def test_parse_idrid_requires_both_label_files(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="Expected exactly one"):
        parse_idrid(tmp_path)


def test_parse_eyeq_drops_absent_images_and_groups_by_patient(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2],
            "image": ["10_left.jpeg", "10_right.jpeg", "99_left.jpeg"],
            "quality": [0, 2, 1],
            "DR_grade": [0, 0, 1],
        }
    ).to_csv(labels, index=False)
    images = tmp_path / "images"
    _touch(images / "10_left.jpeg")
    _touch(images / "10_right.jpeg")

    manifest = parse_eyeq([labels], images)

    assert manifest[QUALITY].tolist() == [0, 2]
    assert manifest[IMAGE_ID].tolist() == ["10_left", "10_right"]
    assert manifest[GROUP].tolist() == ["10", "10"]


def test_parse_eyeq_with_no_images_is_error(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    pd.DataFrame({"image": ["1_left.jpeg"], "quality": [0]}).to_csv(labels, index=False)

    with pytest.raises(DataError, match="None of the"):
        parse_eyeq([labels], tmp_path)


def test_parse_eyepacs_groups_both_eyes_of_a_patient(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()  # the label file is found at any depth
    pd.DataFrame({"image": ["10_left", "10_right", "13_left"], "level": [0, 2, 4]}).to_csv(
        tmp_path / "nested" / "trainLabels.csv", index=False
    )
    for name in ("10_left", "10_right", "13_left"):
        _touch(tmp_path / "images" / f"{name}.jpeg")

    manifest = parse_eyepacs(
        tmp_path,
        "trainLabels.csv",
        Path("images"),
        image_suffix=".jpeg",
        source=Source.EYEPACS,
        id_prefix="",
    )

    assert manifest[GRADE].tolist() == [0, 2, 4]
    assert manifest[GROUP].tolist() == ["10", "10", "13"]
    assert set(manifest[SOURCE]) == {"eyepacs"}
    assert manifest[IMAGE_PATH].iloc[2] == str(tmp_path / "images" / "13_left.jpeg")


def test_parse_eyepacs_missing_image_is_error(tmp_path: Path) -> None:
    pd.DataFrame({"image": ["10_left", "10_right"], "level": [0, 1]}).to_csv(
        tmp_path / "trainLabels.csv", index=False
    )
    _touch(tmp_path / "images" / "10_left.jpeg")

    with pytest.raises(DataError, match="1 of 2 labelled images are missing"):
        parse_eyepacs(
            tmp_path,
            "trainLabels.csv",
            Path("images"),
            image_suffix=".jpeg",
            source=Source.EYEPACS,
            id_prefix="",
        )


def test_parse_eyepacs_test_set_prefixes_ids(tmp_path: Path) -> None:
    pd.DataFrame({"image": ["1_left", "1_right"], "level": [1, 3]}).to_csv(
        tmp_path / "testLabels15.csv", index=False
    )
    for name in ("1_left", "1_right"):
        _touch(tmp_path / "resized test 15" / f"{name}.jpg")

    manifest = parse_eyepacs(
        tmp_path,
        "testLabels15.csv",
        Path("resized test 15"),
        image_suffix=".jpg",
        source=Source.EYEPACS_TEST,
        id_prefix="test15_",
    )

    assert manifest[IMAGE_ID].tolist() == ["test15_1_left", "test15_1_right"]
    assert manifest[GROUP].tolist() == ["test15_1", "test15_1"]
    assert set(manifest[SOURCE]) == {"eyepacs_test"}


def _write_ddr(root: Path, subsets: dict[str, list[tuple[str, int]]]) -> None:
    for subset, rows in subsets.items():
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{subset}.txt").write_text("".join(f"{name} {grade}\n" for name, grade in rows))
        for name, _ in rows:
            _touch(root / subset / name)


def test_parse_ddr_pools_subsets_and_drops_ungradable(tmp_path: Path) -> None:
    _write_ddr(
        tmp_path,
        {
            "train": [("007-0001-000.jpg", 0), ("007-0002-000.jpg", 5)],
            "valid": [("008-0001-000.jpg", 2)],
            "test": [("009-0001-000.jpg", 4)],
        },
    )

    manifest = parse_ddr(tmp_path)

    assert manifest[GRADE].tolist() == [0, 2, 4]
    assert manifest[IMAGE_ID].tolist() == [
        "ddr_007-0001-000",
        "ddr_008-0001-000",
        "ddr_009-0001-000",
    ]
    assert manifest[IMAGE_PATH].iloc[1] == str(tmp_path / "valid" / "008-0001-000.jpg")
    assert set(manifest[SOURCE]) == {"ddr"}


def test_parse_ddr_missing_image_is_error(tmp_path: Path) -> None:
    _write_ddr(tmp_path, {"train": [("a.jpg", 1)], "valid": [], "test": []})
    (tmp_path / "train" / "a.jpg").unlink()

    with pytest.raises(DataError, match="1 of 1 train images are missing"):
        parse_ddr(tmp_path)
