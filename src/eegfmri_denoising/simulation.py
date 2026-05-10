import mne

class sphere_model:
    def __init__(info):
        if info:
            print(info)
        else:
            print('MNE Info not provided, creating sphere with standard parameters...')

