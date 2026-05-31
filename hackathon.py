import mne
import matplotlib.pyplot as plt
import numpy as np

print("works")
raw = mne.io.read_raw(
    r"C:\Users\Peter\OneDrive\Desktop\NeuroHackathon\eegfmri_denoising\hackathon_data\bids_dataset\sub-example\eeg\sub-example_task-rest_eeg.vhdr"
)

print(raw)
# raw.plot(picks = ['O1'])
# plt.show()
data = raw.get_data(picks=["O1"])
CWL_data1 = raw.get_data(picks=["CWL1"])
CWL_data2 = raw.get_data(picks=["CWL2"])
# cwl1, cwl2
print(f"data is{type(data)}\ndata shape is {data.shape}")
reduced_data = data[:, 16000:19000]
reduced_CWL_data1 = CWL_data1[:, 16000:19000]
reduced_CWL_data2 = CWL_data2[:, 16000:19000]

print(f"reduced_data shape is {reduced_data.shape}")
fig, ax = plt.subplots(3)
fig.suptitle("15k-20k data")
ax[0].plot(reduced_data[0, :])
ax[1].plot(reduced_CWL_data1[0, :])
ax[2].plot(reduced_CWL_data2[0, :])
plt.show()
# plt.figure(figsize= (5,8))
# plt.plot(data_x)
# plt.show()

# df = pd.DataFrame({'O1_data':data})
# print(df.head())

# print(data)

# remove gradients func from debug notebook
def remove_gradients(
    raw,
    event_name="Gradient/G  1",
    event_offset=0.0,
    window_length=None,
    baseline_correction=False,
    baseline=None,
):
    """
    Remove gradient artefact from raw M/EEG data using
    average artefact subtraction (AAS), while keeping
    the full continuous recording.
    """

    # -----------------------------
    # Load data
    # -----------------------------`        `
    raw.load_data()
    sfreq = raw.info["sfreq"]

    # -----------------------------
    # Sliding window logic
    # -----------------------------
    sliding_window = True
    if window_length is None:
        print("No window length selected. Using all volumes for template.")
        sliding_window = False
    elif window_length % 2 == 0:
        window_length += 1
        print(f"Window length must be odd. Using {window_length} instead.")

    # -----------------------------
    # Get gradient events
    # -----------------------------
    events, events_id = mne.events_from_annotations(raw)
    if event_name not in events_id:
        raise ValueError(f"{event_name} not found in annotations.")

    grad_events = events[events[:, 2] == events_id[event_name]].copy()
    n_events = len(grad_events)

    if n_events < 2:
        raise ValueError("Need at least 2 events to estimate TR.")

    # -----------------------------
    # Apply offset ONCE (in samples)
    # -----------------------------
    offset_samp = int(round(event_offset * sfreq))
    print(f"OFFSET SAMP = {offset_samp}")
    grad_events[:, 0] += offset_samp

    # -----------------------------
    # Compute TR
    # -----------------------------
    tr_samples = grad_events[1, 0] - grad_events[0, 0]
    tr_sec = tr_samples / sfreq
    print(f"Scanner Repetition Time = {tr_sec:.3f} s")

    # -----------------------------
    # Epoching (now simple)
    # -----------------------------
    epochs = mne.Epochs(
        raw,
        grad_events,
        tmin=0,
        tmax=tr_sec
        - (
            1 / sfreq
        ),  ### TODO - (1 / sfreq) THIS IS VERY IMPORTANT, DO WE DEFO NEED IT? IS IT DOING WHAT WE THINK
        baseline=None,
        preload=True,
        reject_by_annotation=True,
    )

    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = data.shape

    half_win = window_length // 2 if sliding_window else 0

    # -----------------------------
    # Main AAS loop
    # -----------------------------
    for ch in range(n_channels):

        ch_epochs = data[:, ch, :]  # (n_epochs, n_times)

        for i, event in enumerate(grad_events):

            # Template construction
            if not sliding_window:
                noise_avg = ch_epochs.mean(axis=0)
            else:
                w_start = max(0, i - half_win)
                w_stop = min(n_events, i + half_win + 1)
                noise_avg = ch_epochs[w_start:w_stop].mean(axis=0)

            # Optional baseline correction
            if baseline_correction:
                if baseline is not None:
                    b_start = int((baseline[0] / tr_sec) * n_times)
                    b_stop = int((baseline[1] / tr_sec) * n_times)
                    noise_avg -= noise_avg[b_start:b_stop].mean()
                else:
                    noise_avg -= noise_avg.mean()

            # Subtract artefact
            start = event[0]
            stop = start + n_times

            if start < 0 or stop > raw.n_times:
                continue

            raw._data[ch, start:stop] -= noise_avg

    return raw
