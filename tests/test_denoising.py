import numpy as np
import mne
from src.eegfmri_denoising.denoising import remove_gradients

def test_remove_gradients():
    fs = 5000        # sampling rate
    duration = 12   # seconds
    tr = 1.2        # TR in seconds (Ensure not a multiple )
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs

    # 1. Construct EEG signal
    clean = np.sin(2 * np.pi * 9 * t)          # desired signal
    artifact = 5 * np.sin(2 * np.pi * 50 * t)   # known artifact
    data = clean + artifact
    data = data[np.newaxis, :]  # shape (n_channels, n_samples)

    # 2. Create Raw object
    info = mne.create_info(ch_names=['EEG001'], sfreq=fs, ch_types=['eeg'])
    raw = mne.io.RawArray(data, info)

    # 3. Add evenly spaced TR markers as annotations
    tr_samples = np.arange(0, n_samples, int(tr * fs))
    onset_times = tr_samples / fs  # convert sample index to seconds
    durations = np.full_like(onset_times, 0.001)  # tiny duration for each marker
    descriptions = ['TR'] * len(onset_times)

    annotations = mne.Annotations(onset=onset_times,
                                  duration=durations,
                                  description=descriptions)
    raw.set_annotations(annotations)

    # 4. Run denoising
    y_raw = remove_gradients(raw, 'TR', baseline_correction=True)

    # Extract EEG data
    y = y_raw.get_data().flatten()
    
    # Check cleaned signal is closer to clean than raw
    corr_after = np.corrcoef(y, clean)[0, 1]

    assert y.shape == data.shape[1:]
    assert np.isfinite(y).all()
    assert round(corr_after, 10) == 1.0
