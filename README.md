<center>
<h1>EEG-fMRI Denoising</h1>
</center>

<p align="center">
  <a href="https://github.com/psf/black">
    <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
  </a>
  <a href="https://pypi.org/project/eegfmri-denoising/">
    <img alt="PyPI version" src="https://img.shields.io/pypi/v/eegfmri-denoising">
  </a>
  <a href="https://pypi.org/project/eegfmri-denoising/">
    <img alt="Python versions" src="https://img.shields.io/pypi/pyversions/eegfmri-denoising.svg">
  </a>
  <a href="https://dl.circleci.com/status-badge/redirect/gh/Bingram22/eegfmri_denoising/tree/main">
    <img alt="CircleCI" src="https://dl.circleci.com/status-badge/img/gh/Bingram22/eegfmri_denoising/tree/main.svg?style=svg">
  </a>
  <a href="https://codecov.io/gh/Bingram22/eegfmri_denoising">
    <img alt="Coverage" src="https://codecov.io/gh/Bingram22/eegfmri_denoising/branch/main/graph/badge.svg">
  </a>
</p>

## Installation
```
pip install eegfmri_denoising
```
or
```
uv install eegfmri_denoising
```
---
## Current Goals

- [ ] Create reliable examples
- [ ] Create documentation (sphinx)
- [ ] Implement carbon wire loop regression
- [ ] Implement ECG peak detection
- [ ] Implement BCG artifact simulation
- [ ] Update units tests
- [ ] Continuous intergration
- [ ] Implement pooch for fetching example data.
- [ ] Implement QC measures

## Contributing
1. Fork this github repo
2. Clone the fork to your pc
3. Install uv (https://docs.astral.sh/uv/getting-started/installation/)
4. cd to the repo
5. Run "uv sync" to install dependencies

## References
- Allen, P. J., Josephs, O., & Turner, R. (2000). A method for removing imaging artifact from continuous EEG recorded during functional MRI. Neuroimage, 12(2), 230-239.
- Allen, P. J., Polizzi, G., Krakow, K., Fish, D. R., & Lemieux, L. (1998). Identification of EEG events in the MR scanner: the problem of pulse artifact and a method for its subtraction. Neuroimage, 8(3), 229-239.
- Niazy, R. K., Beckmann, C. F., Iannetti, G. D., Brady, J. M., & Smith, S. M. (2005). Removal of FMRI environment artifacts from EEG data using optimal basis sets. Neuroimage, 28(3), 720-737.
- van der Meer, J. N., Pampel, A., Van Someren, E. J., Ramautar, J. R., van der Werf, Y. D., Gomez-Herrero, G., ... & Walter, M. (2016). Carbon-wire loop based artifact correction outperforms post-processing EEG/fMRI corrections—A validation of a real-time simultaneous EEG/fMRI correction method. Neuroimage, 125, 880-894.
- Yan, W. X., Mullinger, K. J., Geirsdottir, G. B., & Bowtell, R. (2010). Physical modeling of pulse artefact sources in simultaneous EEG/fMRI. Human brain mapping, 31(4), 604-620.
- Yan, W. X., Mullinger, K. J., Brookes, M. J., & Bowtell, R. (2009). Understanding gradient artefacts in simultaneous EEG/fMRI. Neuroimage, 46(2), 459-471.
## Example Usage
TBC
