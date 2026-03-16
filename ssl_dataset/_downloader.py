# ssl_dataset/_downloader.py
#
# Handles all downloading from HuggingFace Hub and local cache management.
#
# Design decisions:
#   - Files are cached in ~/.cache/ssl_dataset/ after first download
#   - If a file already exists in cache, it is NEVER re-downloaded
#   - Each sub-library (landmarks, skeleton, preprocessed) has its own
#     cache subfolder so they don't interfere with each other
#   - All three sub-libraries use the same downloader — no duplicated logic

import pathlib
from huggingface_hub import hf_hub_download

from ._constants import (
    HF_REPO_LANDMARKS,
    HF_REPO_SKELETON,
    HF_REPO_PREPROCESSED,
    CACHE_DIR_LANDMARKS,
    CACHE_DIR_SKELETON,
    CACHE_DIR_PREPROCESSED,
)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _download_file(
    repo_id: str,
    remote_path: str,
    local_cache_dir: pathlib.Path,
) -> pathlib.Path:
    """
    Download a single file from a HuggingFace dataset repo into local_cache_dir.
    If the file already exists locally it is returned immediately without
    any network request — this is what makes the library fast after the
    first use.

    Parameters
    ----------
    repo_id : str
        The HuggingFace repo, e.g. "Jayasha/ssl-landmarks"
    remote_path : str
        Path inside the repo, e.g. "static/class_000/0.json"
    local_cache_dir : pathlib.Path
        Root cache folder for this sub-library, e.g.
        ~/.cache/ssl_dataset/landmarks/

    Returns
    -------
    pathlib.Path
        Absolute path to the locally cached file.
    """
    local_path = local_cache_dir / remote_path

    # Already cached — return immediately, no download needed
    if local_path.exists():
        return local_path

    # Create the parent directory if it doesn't exist yet
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: repo_type="dataset" is required — without it HuggingFace
    # looks for a model repo instead of a dataset repo and returns 404.
    hf_hub_download(
        repo_id=repo_id,
        filename=remote_path,
        repo_type="dataset",
        local_dir=str(local_cache_dir),
    )

    return local_path


# ---------------------------------------------------------------------------
# Public downloaders — one per sub-library
# ---------------------------------------------------------------------------

def download_landmark_file(remote_path: str) -> pathlib.Path:
    """
    Download a landmark .json file from HuggingFace and cache it locally.

    Parameters
    ----------
    remote_path : str
        Path inside the ssl-landmarks repo.
        Static example : "static/class_000/0.json"
        Dynamic example: "dynamic/class_037/0.json"

    Returns
    -------
    pathlib.Path
        Path to the cached local file.

    Example
    -------
    >>> from ssl_dataset._downloader import download_landmark_file
    >>> path = download_landmark_file("static/class_000/0.json")
    """
    return _download_file(
        repo_id=HF_REPO_LANDMARKS,
        remote_path=remote_path,
        local_cache_dir=CACHE_DIR_LANDMARKS,
    )


def download_skeleton_file(remote_path: str) -> pathlib.Path:
    """
    Download a skeleton image .png file from HuggingFace and cache it locally.

    Parameters
    ----------
    remote_path : str
        Path inside the ssl-skeleton repo.
        Static example : "static/class_000/0.png"
        Dynamic example: "dynamic/class_037/0/frame_000.png"

    Returns
    -------
    pathlib.Path
        Path to the cached local file.
    """
    return _download_file(
        repo_id=HF_REPO_SKELETON,
        remote_path=remote_path,
        local_cache_dir=CACHE_DIR_SKELETON,
    )


def download_preprocessed_file(remote_path: str) -> pathlib.Path:
    """
    Download a preprocessed .npy file from HuggingFace and cache it locally.

    Parameters
    ----------
    remote_path : str
        Path inside the ssl-preprocessed repo.
        Example: "lstm/X_train.npy"
        Example: "lstm/y_test.npy"

    Returns
    -------
    pathlib.Path
        Path to the cached local file.
    """
    return _download_file(
        repo_id=HF_REPO_PREPROCESSED,
        remote_path=remote_path,
        local_cache_dir=CACHE_DIR_PREPROCESSED,
    )


# ---------------------------------------------------------------------------
# Cache management utilities
# ---------------------------------------------------------------------------

def is_cached(sub_library: str, remote_path: str) -> bool:
    """
    Check whether a specific file is already in the local cache,
    without making any network request.

    Parameters
    ----------
    sub_library : str
        One of "landmarks", "skeleton", "preprocessed"
    remote_path : str
        The same remote_path string you would pass to the downloader.

    Returns
    -------
    bool
        True if the file exists in cache, False otherwise.
    """
    cache_dirs = {
        "landmarks":    CACHE_DIR_LANDMARKS,
        "skeleton":     CACHE_DIR_SKELETON,
        "preprocessed": CACHE_DIR_PREPROCESSED,
    }
    if sub_library not in cache_dirs:
        raise ValueError(
            f"sub_library must be one of {list(cache_dirs.keys())}, "
            f"got {sub_library!r}"
        )
    return (cache_dirs[sub_library] / remote_path).exists()


def clear_cache(sub_library: str = "all") -> None:
    """
    Delete cached files for one or all sub-libraries.
    Useful when you want to force a fresh download.

    Parameters
    ----------
    sub_library : str
        "landmarks", "skeleton", "preprocessed", or "all" (default).

    Example
    -------
    >>> from ssl_dataset._downloader import clear_cache
    >>> clear_cache("landmarks")   # delete only landmark cache
    >>> clear_cache()              # delete everything
    """
    import shutil

    cache_dirs = {
        "landmarks":    CACHE_DIR_LANDMARKS,
        "skeleton":     CACHE_DIR_SKELETON,
        "preprocessed": CACHE_DIR_PREPROCESSED,
    }

    if sub_library == "all":
        targets = list(cache_dirs.values())
    elif sub_library in cache_dirs:
        targets = [cache_dirs[sub_library]]
    else:
        raise ValueError(
            f"sub_library must be one of {list(cache_dirs.keys())} or 'all', "
            f"got {sub_library!r}"
        )

    for target in targets:
        if target.exists():
            shutil.rmtree(target)
            print(f"Cleared cache: {target}")
        else:
            print(f"Nothing to clear: {target}")
