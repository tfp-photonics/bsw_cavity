#!/usr/bin/env python3

from pathlib import Path

import meep as mp
import h5py
import numpy as np
from tqdm import tqdm

from util import dict_to_h5


def main():
    fp = Path("index_delta_sweep_n1019.h5")
    sim_res = 50
    lcen = 0.57
    decay_by = 1e-8
    d_wvg = 0.3
    l_wvg = 1
    dpml = 1.0
    fcen = 1 / lcen
    fwidth = 0.2
    nfreqs = 701
    mp.verbosity(0)

    runs = {}
    with h5py.File(fp, "r") as f:
        for gn, grp in f.items():
            runs[float(gn)] = {k: v for k, v in grp.attrs.items()}
            runs[float(gn)].update({k: np.array(v) for k, v in grp.items()})

    for index_delta, run in tqdm(runs.items(), ncols=80, disable=not mp.am_master()):
        radius = run["radius"]
        design = run["design"]
        n1 = run["n1"]
        n2 = run["n2"]

        cavity_center = dpml + l_wvg + radius
        cell = mp.Vector3(2 * (dpml + radius + 1), 2 * dpml + 2 * radius + 1 + l_wvg)

        sources = [
            mp.Source(
                mp.GaussianSource(frequency=fcen, fwidth=fwidth),
                center=mp.Vector3(0, cavity_center),
                size=mp.Vector3(0, 0),
                component=mp.Ex,
            )
        ]

        n1 = mp.Medium(index=n1)
        n2 = mp.Medium(index=n2)

        mg = mp.MaterialGrid(design.shape, n1, n2)

        geometry = [
            mp.Block(
                center=mp.Vector3(0, cavity_center),
                size=mp.Vector3(2 * radius, 2 * radius),
                material=mg,
            ),
            mp.Sphere(center=mp.Vector3(0, cavity_center), radius=0.4, material=n2),
            mp.Block(
                center=mp.Vector3(0, (dpml + l_wvg) / 2),
                size=mp.Vector3(d_wvg, dpml + l_wvg),
                material=n2,
            ),
        ]

        sim = mp.Simulation(
            cell_size=cell,
            boundary_layers=[mp.PML(dpml)],
            geometry=geometry,
            sources=sources,
            resolution=sim_res,
            default_material=n1,
            geometry_center=mp.Vector3(0, cell.y / 2),
            symmetries=[mp.Mirror(mp.X, phase=-1)],
        )

        # "bulk" ldos
        cc = sources[0].center
        mg.update_weights(np.zeros_like(design))
        dft = sim.add_dft_fields(
            [mp.Ex, mp.Ey],
            fcen,
            fwidth,
            1,
            center=cc,
            size=mp.Vector3(2 * radius, 2 * radius),
        )
        ldos = mp.Ldos(fcen, fwidth, nfreqs)
        mmon = sim.add_mode_monitor(
            fcen,
            fwidth,
            nfreqs,
            mp.ModeRegion(center=mp.Vector3(0, dpml), size=mp.Vector3(cell.x, 0)),
        )
        box_x1 = sim.add_flux(
            fcen,
            fwidth,
            nfreqs,
            mp.FluxRegion(center=cc + mp.Vector3(0.5, 0), size=mp.Vector3(0, 1)),
        )
        box_x2 = sim.add_flux(
            fcen,
            fwidth,
            nfreqs,
            mp.FluxRegion(center=cc + mp.Vector3(-0.5, 0), size=mp.Vector3(0, 1)),
        )
        box_y1 = sim.add_flux(
            fcen,
            fwidth,
            nfreqs,
            mp.FluxRegion(center=cc + mp.Vector3(0, 0.5), size=mp.Vector3(1, 0)),
        )
        box_y2 = sim.add_flux(
            fcen,
            fwidth,
            nfreqs,
            mp.FluxRegion(center=cc + mp.Vector3(0, -0.5), size=mp.Vector3(1, 0)),
        )

        sim.run(
            mp.dft_ldos(ldos=ldos),
            until_after_sources=mp.stop_when_dft_decayed(decay_by),
        )

        run["ldos_data_bulk"] = np.array(sim.ldos_data)
        run["mode_coeffs_bulk"] = sim.get_eigenmode_coefficients(
            mmon, [1], eig_parity=mp.EVEN_Z
        ).alpha
        run["fluxes_bulk"] = np.stack(
            [mp.get_fluxes(box) for box in [box_x1, box_x2, box_y1, box_y2]]
        )

        # design ldos
        sim.reset_meep()
        mg.update_weights(design)
        dft = sim.add_dft_fields(
            [mp.Ex, mp.Ey],
            fcen,
            fwidth,
            1,
            center=cc,
            size=mp.Vector3(2 * radius, 2 * radius),
        )
        hmv = mp.Harminv(mp.Ex, sources[0].center, fcen, fwidth)
        ldos = mp.Ldos(fcen, fwidth, nfreqs)
        mmon = sim.add_mode_monitor(
            fcen,
            fwidth,
            nfreqs,
            mp.ModeRegion(center=mp.Vector3(0, dpml), size=mp.Vector3(cell.x, 0)),
        )
        box_x1 = sim.add_flux(
            fcen,
            fwidth,
            nfreqs,
            mp.FluxRegion(center=cc + mp.Vector3(0.5, 0), size=mp.Vector3(0, 1)),
        )
        box_x2 = sim.add_flux(
            fcen,
            fwidth,
            nfreqs,
            mp.FluxRegion(center=cc + mp.Vector3(-0.5, 0), size=mp.Vector3(0, 1)),
        )
        box_y1 = sim.add_flux(
            fcen,
            fwidth,
            nfreqs,
            mp.FluxRegion(center=cc + mp.Vector3(0, 0.5), size=mp.Vector3(1, 0)),
        )
        box_y2 = sim.add_flux(
            fcen,
            fwidth,
            nfreqs,
            mp.FluxRegion(center=cc + mp.Vector3(0, -0.5), size=mp.Vector3(1, 0)),
        )

        sim.run(
            mp.dft_ldos(ldos=ldos),
            mp.after_sources(hmv),
            until_after_sources=mp.stop_when_dft_decayed(decay_by),
        )

        run["fluxes_design"] = np.stack(
            [mp.get_fluxes(box) for box in [box_x1, box_x2, box_y1, box_y2]]
        )
        run["ldos_freqs"] = np.array(mp.get_ldos_freqs(ldos))
        run["ldos_data_design"] = np.array(sim.ldos_data)
        run["mode_coeffs_design"] = sim.get_eigenmode_coefficients(
            mmon, [1], eig_parity=mp.EVEN_Z
        ).alpha
        run["fields_ex"] = sim.get_dft_array(dft, mp.Ex, 0)
        run["fields_ey"] = sim.get_dft_array(dft, mp.Ey, 0)
        run["harminv"] = {
            m.freq: {"decay": m.decay, "Q": m.Q, "amp": m.amp, "err": m.err}
            for m in hmv.modes
        }
        run["eval_res"] = sim_res

    if mp.am_master():
        with h5py.File(f"{fp.stem}_eval_res{sim_res}.h5", "w") as f:
            dict_to_h5(runs, f)


if __name__ == "__main__":
    main()
