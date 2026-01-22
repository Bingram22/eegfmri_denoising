import mne
import numpy as np


def remove_gradients(
    raw,
    event_name="Gradient/G  1",
    window_length=None,
    baseline_correction=False,
    baseline=None,
):
    """
    Remove gradient artifact from raw M/EEG data while keeping the full recording.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw M/EEG data.
    event_name : str
        Name of the scanner gradient event in annotations.
    window_length : int or None
        Odd number of events to use for sliding-window artifact template.
        If None, uses all events.
    baseline_correction : bool
        Whether to baseline-correct each template before subtraction (default False).
    baseline : tuple or None
        Tuple (tmin, tmax) in seconds relative to each epoch for baseline correction.
        If None, uses the whole template.

    Returns
    -------
    clean_raw : mne.io.RawArray
        Raw data with artifact segments cleaned.
    """
    sliding_window = True

    if window_length is None:
        print(
            "No window length selected. Using all volumes to create template artifact."
        )
        sliding_window = False
    elif window_length % 2 == 0:
        window_length += 1
        print(f"Window length must be odd. Window length is now {window_length}")

    # Get events
    events, events_id = mne.events_from_annotations(raw)
    if event_name not in events_id:
        raise ValueError(f"{event_name} not found in annotations.")

    relevant_events = events[events[:, 2] == events_id[event_name]]

    # Shift start sample by +1 to match BVA
    # relevant_events[:, 0] += 1 # TODO FIX THIS BODGE. MUST BE TODO with MNE ANNOTATIONS THING annat to event thing i mean.
    # print(relevant_events)
    number_of_events = len(relevant_events)

    if number_of_events < 2:
        raise ValueError("Need at least 2 events to compute TR.")

    # Compute TR in seconds
    tr_samples = relevant_events[1][0] - relevant_events[0][0]
    tr_sec = tr_samples / raw.info["sfreq"]
    print(f"Scanner Repetition Time = {tr_sec:.3f} s")

    # Epoch data
    epochs = mne.Epochs(
        raw,
        relevant_events,
        tmin=0,
        tmax=tr_sec - (1 / raw.info["sfreq"]),
        # tmax=(tr_sec / raw.info['sfreq']),
        baseline=None,
        preload=True,
    )
    epochs_data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_channels = len(raw.ch_names)
    n_times = epochs_data.shape[2]

    # Prepare cleaned epochs
    cleaned_epochs = np.empty_like(epochs_data)
    half_win = window_length // 2 if sliding_window else 0

    for ch in range(n_channels):
        for i in range(number_of_events):
            if not sliding_window:
                noise_avg = np.mean(epochs_data[:, ch, :], axis=0)
            else:
                window_start = max(0, i - half_win)
                window_stop = min(number_of_events, i + half_win + 1)

                window_size = window_stop - window_start
                print(
                    f"i={i:3d} | window=[{window_start}:{window_stop}) "
                    f"| n_epochs={window_size}"
                )

                noise_avg = np.mean(
                    epochs_data[window_start:window_stop, ch, :], axis=0
                )

            # Baseline correction if requested
            if baseline_correction:
                if baseline is not None:
                    tmin_idx = int((baseline[0] / tr_sec) * n_times)
                    tmax_idx = int((baseline[1] / tr_sec) * n_times)
                    noise_avg -= np.mean(noise_avg[tmin_idx:tmax_idx])
                else:
                    noise_avg -= np.mean(noise_avg)

            # Subtract template
            cleaned_epochs[i, ch, :] = epochs_data[i, ch, :] - noise_avg

    # Insert cleaned epochs back into full raw data
    cleaned_data = raw.get_data().copy()
    for i, event in enumerate(relevant_events):
        start = event[0]
        stop = start + n_times
        cleaned_data[:, start:stop] = cleaned_epochs[i]

    # Return as RawArray
    clean_raw = mne.io.RawArray(cleaned_data, raw.info)
    return clean_raw, cleaned_epochs
