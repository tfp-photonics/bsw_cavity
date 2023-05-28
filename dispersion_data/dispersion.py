#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def main():
    fp_hi = "221022_600_grt1_measured_n_sub0_dispersion.mat"
    fp_lo = "disp_bsw_bare.mat"
    data_lo = loadmat(fp_lo)
    data_hi = loadmat(fp_hi)
    wlen_lo = data_lo["wv_bsw"].squeeze()
    wlen_hi = data_hi["wv_bsw"].squeeze()
    neff_lo = data_lo["n_eff_bsw"].squeeze()
    neff_hi = data_hi["n_eff_bsw"].squeeze()

    print(np.around(neff_lo[np.where(np.abs(wlen_lo - 632) <= 1e-1)], 5))
    print(np.around(neff_hi[np.where(np.abs(wlen_hi - 632) <= 1e-1)], 5))

    plt.scatter(wlen_lo, neff_lo, label="Bare")
    plt.scatter(wlen_hi, neff_hi, label="Coated")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
