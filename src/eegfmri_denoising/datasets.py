"""
Module eegfmri_denoising/datasets.py
"""
from importlib import resources
import pooch
from . import __version__

DATASET = pooch.create(
    path=pooch.os_cache("eegfmri_denoising"),
    base_url="https://gin.g-node.org/Bingram/example_eeg_fmri_dataset/raw/master/bids/",
    version=__version__,
    version_dev="master",
    env="EEGFMRI_DATA_DIR",
    registry=None,
)

DATASET.load_registry(
    resources.files("eegfmri_denoising").joinpath("registry.txt").open()
)

def fetch_example_data():
    """
    Download and cache the example resting state EEG-fMRI dataset.
    Returns the path to the local BIDS dataset root.
    """
    for fname in DATASET.registry:
        DATASET.fetch(fname, progressbar=True)

    return pooch.os_cache("eegfmri_denoising")

if __name__ == "__main__":
    import mne
    print("Downloading example data...")
    bids_root = fetch_example_data()
    print(f"Data cached at: {bids_root}")
