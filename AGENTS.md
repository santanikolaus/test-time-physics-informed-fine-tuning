# Project Instructions

## Stack
- language: Python, venv
- framework: PyTorch, Lightning
- test runner: pytest

### master thesis
phase 1: reproduce PINO paper table 1,8 Kolmogorov, Darcy flow with physics aware FNO-operator (PINO)
phase 2: build ood detection mechanism.
phase 3: act on ood trigger adaption mechanism.
### files: always read
documentation/paths.yaml
documentation/setup/server.md

### files: only read when necessary/asked to
configs/experiments — most recent experiment configs
documentation/ — experiment logs, journey files, OOD analysis docs, and operator evaluation guides
scripts/ — analysis and inference scripts for OOD experiments (residual sweeps, spectral distances, ROC, ν-sweep)
src/ — core library: solvers, datasets, PDE losses, Lightning modules, FNO model
