# Inverse design of cavities for Bloch Surface Waves interfaced to integrated waveguides

This repository contains code for the paper "Inverse design of cavities for Bloch Surface Waves interfaced to integrated waveguides" ([PNFA](https://doi.org/10.1016/j.photonics.2022.101079)).

## Installation

To run the code, set up a conda environment from the provided environment file and activate it:

```bash
conda env create -f environment.yml -p your_env_prefix
conda activate your_env_prefix
```

## Directory structure

The main inverse design scripts are found under [`optimization/`](optimization/), one for the index delta sweep and one for the dispersion contained in [`dispersion_data/`](dispersion_data/).
The folder [`slurm/`](slurm/) contains Slurm batch scripts for running the 2D optimizations as well as the 3D comparison.
Scripts for data evaluation and plotting are found under [`evaluation/`](evaluation/).

## Citing

If you use this code or associated data for your research, please cite:

```
@article{augenstein2022inverse,
  title = {Inverse design of cavities for Bloch Surface Waves interfaced to integrated waveguides},
  author = {Augenstein, Yannick and Matthieu, Roussey and Thierry, Grosjean and Descrovi, Emiliano and Rockstuhl, Carsten},
  year = 2022,
  journal = {PNFA},
  volume = 52,
  issn = {1569-4410},
  pages = {101079},
  doi = {10.1016/j.photonics.2022.101079}
}
```