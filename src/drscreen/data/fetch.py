"""Download datasets and build their manifests.

Kaggle downloads go through ``kagglehub``. Inside a Kaggle notebook,
``kagglehub`` returns the path of the already-attached input instead of
downloading, so the same code serves local machines and Kaggle sessions.
"""

import logging
from pathlib import Path

import kagglehub
import pandas as pd
import requests
from kagglehub import exceptions as kaggle_errors

from drscreen.data.schema import GRADE, IMAGE_ID, QUALITY, SPLIT, Source, Split
from drscreen.data.settings import DataConfig
from drscreen.data.sources import (
    IDRID_TRAINING_PREFIX,
    parse_aptos,
    parse_ddr,
    parse_eyepacs,
    parse_eyeq,
    parse_idrid,
)
from drscreen.data.splits import assign_splits, cap_per_class
from drscreen.exceptions import DataError

logger = logging.getLogger(__name__)

KAGGLE_CREDENTIALS_HELP = (
    "Create an API token at https://www.kaggle.com/settings and save it as "
    "~/.kaggle/kaggle.json, or set KAGGLE_USERNAME and KAGGLE_KEY."
)
DOWNLOAD_CHUNK_BYTES = 1 << 20
EYEQ_IMAGES_SUBDIR = "images"
EYEQ_LABELS_SUBDIR = "labels"


def download_kaggle_dataset(handle: str, output_dir: Path) -> Path:
    """Download a Kaggle dataset, or locate it when attached to a Kaggle notebook.

    Args:
        handle: ``owner/dataset/versions/N``.
        output_dir: Local destination; ignored inside Kaggle notebooks.

    Returns:
        Directory containing the dataset files.

    Raises:
        DataError: If credentials are missing or the download fails.
    """
    logger.info("Fetching Kaggle dataset %s", handle)
    try:
        return Path(kagglehub.dataset_download(handle, output_dir=str(output_dir)))
    except (kaggle_errors.CredentialError, kaggle_errors.UnauthenticatedError) as exc:
        raise DataError(f"Kaggle credentials missing. {KAGGLE_CREDENTIALS_HELP}") from exc
    except (
        kaggle_errors.KaggleApiHTTPError,
        kaggle_errors.NotFoundError,
        kaggle_errors.BackendError,
        kaggle_errors.DataCorruptionError,
    ) as exc:
        raise DataError(f"Downloading Kaggle dataset {handle} failed: {exc}") from exc


def download_kaggle_competition(competition: str, output_dir: Path) -> Path:
    """Download a Kaggle competition's data, or locate it inside a Kaggle notebook.

    Args:
        competition: Competition slug.
        output_dir: Local destination; ignored inside Kaggle notebooks.

    Returns:
        Directory containing the competition files.

    Raises:
        DataError: If credentials are missing, the competition rules have not
            been accepted, or the download fails.
    """
    logger.info("Fetching Kaggle competition %s", competition)
    try:
        return Path(kagglehub.competition_download(competition, output_dir=str(output_dir)))
    except (kaggle_errors.CredentialError, kaggle_errors.UnauthenticatedError) as exc:
        raise DataError(f"Kaggle credentials missing. {KAGGLE_CREDENTIALS_HELP}") from exc
    except kaggle_errors.KaggleApiHTTPError as exc:
        raise DataError(
            f"Downloading competition {competition} failed: {exc}. If this is a 403, accept "
            f"the rules at https://www.kaggle.com/competitions/{competition}/rules first."
        ) from exc
    except (
        kaggle_errors.NotFoundError,
        kaggle_errors.BackendError,
        kaggle_errors.DataCorruptionError,
    ) as exc:
        raise DataError(f"Downloading competition {competition} failed: {exc}") from exc


def download_file(url: str, destination: Path, timeout_seconds: float) -> Path:
    """Download a file over HTTP unless it already exists.

    The file is written to a temporary name and renamed on success, so an
    interrupted download never leaves a truncated file at ``destination``.

    Args:
        url: Source URL.
        destination: Local file path.
        timeout_seconds: Connect and read timeout.

    Returns:
        ``destination``.

    Raises:
        DataError: If the request fails or returns an HTTP error status.
    """
    if destination.is_file():
        logger.info("Using cached %s", destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    logger.info("Downloading %s", url)
    try:
        with requests.get(url, stream=True, timeout=timeout_seconds) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(DOWNLOAD_CHUNK_BYTES):
                    handle.write(chunk)
    except requests.RequestException as exc:
        partial.unlink(missing_ok=True)
        raise DataError(f"Downloading {url} failed: {exc}") from exc
    partial.rename(destination)
    return destination


def fetch_manifest(source: Source, config: DataConfig) -> pd.DataFrame:
    """Download one dataset and build its manifest with split assignments.

    APTOS, EyePACS and EyeQ are split into train and validation. IDRiD is
    marked entirely as external and is never used for training or model
    selection, unless ``sources.idrid.adaptation`` is set: then its official
    training subset is split into train and validation and only its official
    test subset is external. EyePACS is capped per grade before splitting.

    Args:
        source: Dataset to fetch.
        config: Data pipeline configuration.

    Returns:
        The manifest, including a ``split`` column.

    Raises:
        DataError: If downloading or parsing fails.
    """
    raw_dir = config.paths.raw_dir / source.value
    sources = config.sources
    if source is Source.APTOS:
        root = download_kaggle_competition(sources.aptos.kaggle_competition, raw_dir)
        return assign_splits(parse_aptos(root), GRADE, config.split.val_fraction, config.split.seed)
    if source is Source.IDRID:
        manifest = parse_idrid(download_kaggle_dataset(sources.idrid.kaggle_dataset, raw_dir))
        if not sources.idrid.adaptation:
            manifest[SPLIT] = Split.EXTERNAL.value
            return manifest
        official_train = manifest[IMAGE_ID].str.startswith(IDRID_TRAINING_PREFIX)
        adaptation = assign_splits(
            manifest[official_train], GRADE, config.split.val_fraction, config.split.seed
        )
        test = manifest[~official_train].assign(**{SPLIT: Split.EXTERNAL.value})
        return pd.concat([adaptation, test], ignore_index=True)
    if source in (Source.EYEPACS, Source.EYEPACS_TEST):
        eyepacs = sources.eyepacs if source is Source.EYEPACS else sources.eyepacs_test
        root = download_kaggle_dataset(eyepacs.kaggle_dataset, raw_dir)
        parsed = parse_eyepacs(
            root,
            eyepacs.labels_file,
            eyepacs.images_subdir,
            image_suffix=eyepacs.image_suffix,
            source=source,
            id_prefix=eyepacs.id_prefix,
        )
        manifest = cap_per_class(parsed, GRADE, eyepacs.max_images_per_grade, config.split.seed)
        return assign_splits(manifest, GRADE, config.split.val_fraction, config.split.seed)
    if source is Source.DDR:
        root = download_kaggle_dataset(sources.ddr.kaggle_dataset, raw_dir)
        manifest = cap_per_class(
            parse_ddr(root / sources.ddr.grading_subdir),
            GRADE,
            sources.ddr.max_images_per_grade,
            config.split.seed,
        )
        return assign_splits(manifest, GRADE, config.split.val_fraction, config.split.seed)
    images_root = download_kaggle_dataset(sources.eyeq.kaggle_dataset, raw_dir / EYEQ_IMAGES_SUBDIR)
    label_files = [
        download_file(
            str(url),
            raw_dir / EYEQ_LABELS_SUBDIR / Path(url.path or "").name,
            config.download_timeout_seconds,
        )
        for url in sources.eyeq.label_urls
    ]
    manifest = parse_eyeq(label_files, images_root / sources.eyeq.images_subdir)
    return assign_splits(manifest, QUALITY, config.split.val_fraction, config.split.seed)
