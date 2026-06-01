import numpy as np
import mne
from src.eegfmri_denoising.quality_assurance import get_rms  # adjust module name


def test_get_rms():
    amplitude = 3.0
    sfreq = 1000.0
    duration = 10.0
    n_samples = int(duration * sfreq)
    t = np.arange(n_samples) / sfreq
    data = amplitude * np.sin(2 * np.pi * 10 * t)[np.newaxis, :]

    info = mne.create_info(ch_names=["EEG001"], sfreq=sfreq, ch_types=["eeg"])
    raw = mne.io.RawArray(data, info)

    event_times = np.linspace(0, duration - 2.0, 5)
    annotations = mne.Annotations(
        onset=event_times, duration=np.full(5, 0.001), description=["Gradient/G  1"] * 5
    )
    raw.set_annotations(annotations)

    expected_rms = amplitude / np.sqrt(2)
    result = get_rms(raw)
    assert np.isclose(result, expected_rms, rtol=0.05)
