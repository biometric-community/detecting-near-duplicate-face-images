# Deviations — Detecting Near-Duplicate Face Images

| ID | Paper | Ours | Justification |
|----|-------|------|---------------|
| D1 | MATLAB Enhanced PRNU (Li 2010 mex) | Gaussian high-pass residual (`prnu_residual`) | Filter/ mex not portable; residual keeps sensor-noise-like high-freq cue for link prediction |
| D2 | NDFI / CelebAHQ-FM / MORPH test sets | Held-out LFW-synthesized IPTs | Official NDFI not under `projects/datasets/`; LFW is the paper’s training source |
| D3 | TensorFlow GCN/ChebNet (tkipf) | PyTorch ChebNet (`neardup/models/chebnet.py`) | Skill requires PyTorch; same Chebyshev K=3, hidden=16, Adam 1e-2 |
| D4 | Precomputed MATLAB mat feature graphs | On-the-fly IPT synthesis from LFW with Table 1 transforms | Reproducible without proprietary TRAININGSET folders |
| D5 | Full IPF spectral clustering stage | Single-IPT reconstruction eval | Focus on ChebNet+PRNU core; IPF clustering left as future work |
| D6 | Weight decay on TF vars (sum) | Mean L2 on ChebConv1 | Stabilizes training with 96² pixel features |
