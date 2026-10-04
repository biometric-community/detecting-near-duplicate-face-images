# Source code

Upstream: https://github.com/sudban3089/DetectingNear-Duplicates

- TensorFlow ChebNet/GCN (`Nodeembedding.py`, `models.py`, `layers.py`, `utils.py`) — refactored to PyTorch in `neardup/`.
- MATLAB Enhanced PRNU (`Filter/`, `Functions/`, `PhaseNoiseExtract*.m`) — replaced by a differentiable/Python PRNU residual proxy; see `DEVIATIONS.md`.
- Reference copies of TF helpers under `upstream/` (not executed).
