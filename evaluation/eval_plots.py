#!/usr/bin/env python3

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from util import h5_to_dict

runs = h5_to_dict(h5py.File("index_delta_sweep_n1019_eval_res50.h5", "r"), {})
rr = runs[list(runs.keys())[0]]["radius"]
cf = len(runs[list(runs.keys())[0]]["ldos_freqs"]) // 2
cx, cy = np.array(runs[list(runs.keys())[0]]["fields_ex"].shape) // 2

nx = ny = int(np.around(np.sqrt(len(runs))))
if len(runs) % ny != 0:
    ny += 1

extent = (-rr, rr, -rr - 1, rr)

fig, ax = plt.subplots(2, 4, figsize=(6.3, 3), sharex=True, sharey=True)
for axi, run in zip(
    np.array(ax).T, [runs[k] for k in ["0.01", "0.05", "0.09", "0.13"]]
):
    axi[0].set_title(rf"$\Delta n = {np.around(run['index_delta'], 3)}$", y=1.02)
    field = np.abs(run["fields_ex"]) ** 2 + np.abs(run["fields_ey"]) ** 2
    img = axi[1].imshow(
        field.T, origin="lower", cmap="magma", norm=LogNorm(1e-6, 1e6), extent=extent
    )
    design = run["permittivity"]
    axi[0].imshow(design.T, origin="lower", cmap="gray_r", extent=extent)
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.95, 0.24, 0.025, 0.6])
cbar_ax.set_title(r"$|E|^2$", x=1.2, y=1.02)
fig.colorbar(img, cax=cbar_ax)
ax[0][0].set_ylabel(r"y (\si{\um})")
ax[1][0].set_ylabel(r"y (\si{\um})")
ax[1][0].set_xlabel(r"x (\si{\um})")
ax[1][1].set_xlabel(r"x (\si{\um})")
ax[1][2].set_xlabel(r"x (\si{\um})")
ax[1][3].set_xlabel(r"x (\si{\um})")
fig.subplots_adjust(hspace=0.0, wspace=0.2, top=0.97)
fig.savefig(
    "paper/figures/fields_designs.pdf", bbox_inches="tight", pad_inches=0, dpi=1000
)

# fig, ax = plt.subplots(1, 2, figsize=(6.3, 3.4))
# color = plt.cm.turbo(np.linspace(1, 0, len(runs)))
# for axi in ax:
#     axi.set_xlabel(r"$\lambda$ (\si{\nm})")
#     axi.xaxis.set_ticks(np.linspace(540, 600, 5))
#     axi.axvline(570, ls="--", color="gray", alpha=0.5)
# for idx, (k, run) in enumerate(runs.items()):
#     ax[0].plot(
#         1000 / run["ldos_freqs"],
#         np.sum(np.abs(run["fluxes_design"]), axis=0)
#         / np.sum(np.abs(run["fluxes_bulk"]), axis=0),
#         color=color[idx],
#         label=k,
#     )
#     ax[1].plot(
#         1000 / run["ldos_freqs"],
#         np.abs(run["mode_coeffs_design"][0, :, 1]) ** 2
#         / np.sum(np.abs(run["fluxes_design"]), axis=0),
#         color=color[idx],
#     )
# ax[0].set_yscale("log")
# ax[0].set_ylim([1e-1, 1.3e2])
# ax[0].set_ylabel("Purcell enhancement")
# ax[1].set_ylabel("Coupling efficiency")
# handles, labels = ax[0].get_legend_handles_labels()
# ax[1].legend(
#     handles[::-1],
#     labels[::-1],
#     title=r"$\Delta n$",
#     ncol=1,
#     loc=0,
#     bbox_to_anchor=(1.04, 1.01),
# )
# fig.subplots_adjust(wspace=0.3)
# fig.savefig(
#     "paper/figures/purcell_coupling_spectra.pdf",
#     bbox_inches="tight",
#     pad_inches=0,
#     dpi=1000,
# )

# dn = [float(k) for k in runs.keys()]
# purcell = [
#     np.sum(np.abs(run["fluxes_design"]), axis=0)[cf]
#     / np.sum(np.abs(run["fluxes_bulk"]), axis=0)[cf]
#     for run in runs.values()
# ]
# coupling = [
#     np.abs(run["mode_coeffs_design"][0, cf, 1]) ** 2
#     / np.sum(np.abs(run["fluxes_design"]), axis=0)[cf]
#     for run in runs.values()
# ]
# fig, ax1 = plt.subplots(1, 1, figsize=(3.0, 2))
# color = "tab:blue"
# ax1.plot(dn, purcell, ls="--", marker="x", lw=1, color=color)
# ax1.set_xlabel(r"$\Delta n$")
# ax1.set_ylabel("Purcell enhancement", color=color)
# ax1.set_ylim(0, 80)
# ax1.tick_params(axis="y", labelcolor=color)
# ax1.xaxis.set_ticks(np.linspace(dn[0], dn[-1], 5))
# ax1.yaxis.set_ticks(np.linspace(0, 80, 5))
# color = "tab:red"
# ax2 = ax1.twinx()
# ax2.set_ylim(0.1, 0.6)
# ax2.yaxis.set_ticks(np.linspace(0.1, 0.6, 6))
# ax2.plot(dn, coupling, ls="--", marker="x", lw=1, color=color)
# ax2.set_ylabel("Coupling efficiency", color=color)
# ax2.tick_params(axis="y", labelcolor=color)
# fig.tight_layout()
# fig.savefig(
#     "paper/figures/purcell_coupling_570nm.pdf",
#     bbox_inches="tight",
#     pad_inches=0,
#     dpi=1000,
# )

# data_3d = h5_to_dict(h5py.File("eval_3d_res80_n1019_nomode_dn0130.h5", "r"), {})
# run = runs["0.13"]

# # fields_2d = np.abs(run["fields_ex"]) ** 2 + np.abs(run["fields_ey"]) ** 2
# # fields_3d = np.sum(np.abs(data_3d["fields_plane_xy_dr"] ** 2), axis=0)
# # fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
# # im1 = ax[0].imshow(
# #     fields_2d.T, norm=LogNorm(1e-2, 1e5), origin="lower", cmap="magma", extent=extent
# # )
# # ax[1].imshow(
# #     fields_3d.T, norm=LogNorm(1e-2, 1e5), origin="lower", cmap="magma", extent=extent
# # )
# # for axi in ax:
# #     axi.set_xticks([-5, 0, 5])
# #     axi.set_yticks([-5, 0, 5])
# #     axi.set_xlabel(r"x (\si{\um})")
# #     axi.set_ylabel(r"y (\si{\um})")
# # fig.subplots_adjust(right=0.9)
# # cbar_ax = fig.add_axes([0.95, 0.365, 0.025, 0.25])
# # cbar_ax.set_title(r"$|E|^2$", x=1.2, y=1.02)
# # fig.colorbar(im1, cax=cbar_ax)
# # fig.savefig("fields_2d_3d.pdf", bbox_inches="tight", pad_inches=0, dpi=1000)

# freqs = data_3d["ldos_freqs"]
# fluxes_bulk_3d = np.sum(np.abs(data_3d["fluxes_bulk"]), axis=0)
# fluxes_design_3d = np.sum(np.abs(data_3d["fluxes_design"]), axis=0)
# purcell_2d = np.sum(np.abs(run["fluxes_design"]), axis=0) / np.sum(
#     np.abs(run["fluxes_bulk"]), axis=0
# )
# purcell_3d = fluxes_design_3d / fluxes_bulk_3d
# fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.2))
# ax[0].plot(1000 / freqs, purcell_2d, "tab:blue")
# ax[0].plot(1000 / freqs, np.roll(purcell_3d, -4), "tab:red")
# ax[0].set_yscale("log")
# ax[0].set_ylim([1e-1, 1e2])
# ax[0].set_ylabel("Purcell enhancement")
# ax[1].plot(
#     1000 / freqs,
#     np.abs(run["mode_coeffs_design"][0, :, 1]) ** 2
#     / np.sum(np.abs(run["fluxes_design"]), axis=0),
#     "tab:blue",
#     label="2D",
# )
# ax[1].plot(
#     1000 / freqs,
#     np.roll(np.abs(data_3d["wvg_flux"] / fluxes_design_3d), -4),
#     "tab:red",
#     label="3D",
# )
# ax[1].set_ylabel("Coupling efficiency")
# for axi in ax:
#     axi.xaxis.set_ticks(np.linspace(540, 600, 5))
#     axi.axvline(570, color="gray", ls="--", alpha=0.5)
#     axi.tick_params(axis="both")
#     axi.set_xlabel(r"$\lambda$ (nm)")
# fig.legend(ncol=2, loc=1, bbox_to_anchor=(0.9, 1.05))
# fig.subplots_adjust(wspace=0.3)
# fig.savefig(
#     "paper/figures/comparison_2d_3d.pdf", bbox_inches="tight", pad_inches=0, dpi=1000
# )
