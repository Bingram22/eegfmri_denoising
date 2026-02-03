import mne
import numpy as np

def get_rms(raw,
    event_name="Gradient/G  1",):
    """
    Compute RMS of gradient-locked residuals.

    Parameters
    ----------
    epochs : np.ndarray
        Shape (n_epochs, n_channels, n_times)

    Returns
    -------
    rms : float
        Mean RMS across channels and epochs
    """
    raw.load_data()

    events, events_id = mne.events_from_annotations(raw)
    if event_name not in events_id:
        raise ValueError(f"{event_name} not found in annotations.")

    relevant_events = events[events[:, 2] == events_id[event_name]]
    number_of_events = len(relevant_events)

    if number_of_events < 2:
        raise ValueError("Need at least 2 events to compute TR.")

    # Compute TR
    ### TODO create this into a helper function
    tr_samples = relevant_events[1][0] - relevant_events[0][0]
    sfreq = raw.info["sfreq"]
    tr_sec = tr_samples / sfreq
    print(f"Scanner Repetition Time = {tr_sec:.3f} s")

    epochs = mne.Epochs(
        raw,
        relevant_events,
        tmin=0,
        tmax=1.5,
        baseline=None,
        preload=True,
    )

    # RMS per epoch, per channel
    rms = np.sqrt(np.mean(epochs.get_data() ** 2, axis=-1))  # (n_epochs, n_channels)

    # Average across epochs and channels
    return np.mean(rms)