import mne

def remove_gradients(
    raw,
    event_name="Gradient/G  1",
    window_length=None,
    baseline_correction=False,
    baseline=None,
):
    """
    Remove gradient artifact from raw M/EEG data while keeping the full recording.
    """
    ### Load data + parse args
    raw = raw.copy()
    raw.load_data()

    sliding_window = True

    if window_length is None:
        print(
            "No window length selected. Using all volumes to create template artifact."
        )
        sliding_window = False
    elif window_length % 2 == 0:
        window_length += 1
        print(f"Window length must be odd. Window length is now {window_length}")

    # get events

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

    # Epoch

    epochs = mne.Epochs(
        raw,
        relevant_events,
        tmin=0,
        tmax=tr_sec - (1 / sfreq),
        baseline=None,
        preload=True,
    )

    n_channels = len(raw.ch_names)
    n_times = epochs.get_data(picks=[0]).shape[-1]

    half_win = window_length // 2 if sliding_window else 0

    for ch in range(n_channels):
        ch_epochs = epochs.get_data(picks=[ch])[:, 0, :]  # (n_epochs, n_times)

        for i, event in enumerate(relevant_events):
            if not sliding_window:
                noise_avg = ch_epochs.mean(axis=0)
            else:
                window_start = max(0, i - half_win)
                window_stop = min(number_of_events, i + half_win + 1)
                noise_avg = ch_epochs[window_start:window_stop].mean(axis=0)

            # Baseline correction
            if baseline_correction:
                if baseline is not None:
                    tmin_idx = int((baseline[0] / tr_sec) * n_times)
                    tmax_idx = int((baseline[1] / tr_sec) * n_times)
                    noise_avg -= noise_avg[tmin_idx:tmax_idx].mean()
                else:
                    noise_avg -= noise_avg.mean()

            # Subtract IN PLACE
            start = event[0]
            stop = start + n_times
            raw._data[ch, start:stop] -= noise_avg

    return raw


def r_peak_detection(raw):
    out = raw.copy()
    return
