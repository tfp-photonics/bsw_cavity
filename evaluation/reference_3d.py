#!/usr/bin/env python3

import meep as mp
import h5py
import numpy as np

from util import dict_to_h5, h5_to_dict


def bulk_flux_pmma():
    sim_res = 80
    decay_by = 1e-8
    lcen = 0.57
    fcen = 1 / lcen
    fwidth = 0.2
    nfreqs = 701
    dpml = lcen
    d_pmma = 0.075

    pmma = mp.Medium(index=1.48)
    cell = mp.Vector3(2 * dpml + 1.0, 2 * dpml + 1.0, 2 * dpml + 1.0)

    sources = [
        mp.Source(
            mp.GaussianSource(frequency=fcen, fwidth=fwidth),
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(0, 0, 0),
            component=mp.Ex,
        )
    ]

    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(dpml)],
        sources=sources,
        resolution=sim_res,
        default_material=pmma,
        symmetries=[mp.Mirror(mp.X, phase=-1)],
    )

    flux_planes = []
    for offset in [-0.5, 0.5]:
        for center, size in [
            [(offset, 0, 0), (0, 1, d_pmma)],
            [(0, offset, 0), (1, 0, d_pmma)],
            [(0, 0, np.sign(offset) * d_pmma / 2), (1, 1, 0)],
        ]:
            flux_planes.append(
                sim.add_flux(
                    fcen,
                    fwidth,
                    nfreqs,
                    mp.FluxRegion(
                        center=mp.Vector3(*center),
                        size=mp.Vector3(*size),
                    ),
                )
            )

    sim.run(until_after_sources=mp.stop_when_dft_decayed(decay_by))

    return np.stack([mp.get_fluxes(plane) for plane in flux_planes])


def main():
    # runs = h5_to_dict(h5py.File("index_delta_sweep_eval_res50.h5", "r"))
    runs = h5_to_dict(h5py.File("index_delta_sweep_n1019.h5", "r"))
    # index_delta = "0.040"
    # index_delta = "0.100"
    index_delta = "0.130"
    # index_delta = "0.090"
    design = runs[index_delta]["design"]
    # radius = runs[index_delta]["radius"]
    radius = 5.7

    sim_res = 80
    lcen = 0.57
    decay_by = 1e-8
    fcen = 1 / lcen
    fwidth = 0.2
    nfreqs = 701

    n_layers = 10

    dpml = lcen
    l_wvg = 1.0 + dpml
    d_wvg = 0.3
    d_substrate = dpml + 0.5
    d_sio2_stack = 0.137
    d_sio2_top = 0.127
    d_ta2o5 = 0.095
    d_pmma = 0.075
    d_air = 1.0 + dpml

    air = mp.Medium(index=1.0)
    glass = mp.Medium(index=1.5)
    sio2 = mp.Medium(index=1.46)
    ta2o5 = mp.Medium(index=2.08)
    pmma = mp.Medium(index=1.48)

    mg = mp.MaterialGrid(design.shape, air, pmma)
    mg.update_weights(design)

    pmma_z = (
        d_substrate
        + n_layers * (d_ta2o5 + d_sio2_stack)
        + d_ta2o5
        + d_sio2_top
        + d_pmma / 2
    )

    cell = mp.Vector3(
        2 * (dpml + radius + 0.5),
        2 * radius + 0.5 + l_wvg + dpml,
        d_substrate
        + n_layers * (d_ta2o5 + d_sio2_stack)
        + d_ta2o5
        + d_sio2_top
        + d_pmma
        + d_air,
    )

    cavity_center = mp.Vector3(0, (l_wvg - dpml - 0.5) / 2, pmma_z)

    xy_pmma = mp.Volume(
        center=mp.Vector3(
            0,
            0,
            d_substrate
            + n_layers * (d_ta2o5 + d_sio2_stack)
            + d_ta2o5
            + d_sio2_top
            + d_pmma / 2,
        ),
        size=mp.Vector3(cell.x, cell.y, 0),
    )
    xy_dr = mp.Volume(
        center=cavity_center,
        size=mp.Vector3(2 * radius, 2 * radius, 0),
    )
    xz = mp.Volume(
        center=mp.Vector3(0, cavity_center.y, cell.z / 2),
        size=mp.Vector3(cell.x, 0, cell.z),
    )
    xz_wvg = mp.Volume(
        center=mp.Vector3(0, -cell.y / 2 + dpml, cell.z / 2),
        size=mp.Vector3(cell.x, 0, cell.z),
    )
    yz = mp.Volume(
        center=mp.Vector3(0, 0, cell.z / 2),
        size=mp.Vector3(0, cell.y, cell.z),
    )

    geometry = [
        mp.Block(
            center=mp.Vector3(0, 0, d_substrate / 2),
            size=mp.Vector3(mp.inf, mp.inf, d_substrate),
            material=glass,
        )
    ]
    for idx in range(n_layers):
        z = d_substrate + idx * (d_ta2o5 + d_sio2_stack)
        geometry.extend(
            [
                mp.Block(
                    center=mp.Vector3(0, 0, z + d_ta2o5 / 2),
                    size=mp.Vector3(mp.inf, mp.inf, d_ta2o5),
                    material=ta2o5,
                ),
                mp.Block(
                    center=mp.Vector3(0, 0, z + d_ta2o5 + d_sio2_stack / 2),
                    size=mp.Vector3(mp.inf, mp.inf, d_sio2_stack),
                    material=sio2,
                ),
            ]
        )
    geometry.extend(
        [
            mp.Block(
                center=mp.Vector3(
                    0,
                    0,
                    d_substrate + n_layers * (d_ta2o5 + d_sio2_stack) + d_ta2o5 / 2,
                ),
                size=mp.Vector3(mp.inf, mp.inf, d_ta2o5),
                material=ta2o5,
            ),
            mp.Block(
                center=mp.Vector3(
                    0,
                    0,
                    d_substrate
                    + n_layers * (d_ta2o5 + d_sio2_stack)
                    + d_ta2o5
                    + d_sio2_top / 2,
                ),
                size=mp.Vector3(mp.inf, mp.inf, d_sio2_top),
                material=sio2,
            ),
            mp.Block(
                center=cavity_center,
                size=mp.Vector3(2 * radius, 2 * radius, d_pmma),
                material=mg,
            ),
            mp.Cylinder(
                center=cavity_center,
                radius=0.4,
                height=d_pmma,
                material=pmma,
            ),
            mp.Block(
                center=mp.Vector3(0, -cell.y / 2 + l_wvg / 2, pmma_z),
                size=mp.Vector3(d_wvg, l_wvg, d_pmma),
                material=pmma,
            ),
        ]
    )

    sources = [
        mp.Source(
            mp.GaussianSource(frequency=fcen, fwidth=fwidth),
            center=cavity_center,
            size=mp.Vector3(0, 0, 0),
            component=mp.Ex,
        )
    ]

    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(dpml)],
        geometry=geometry,
        sources=sources,
        resolution=sim_res,
        geometry_center=mp.Vector3(0, 0, cell.z / 2),
        symmetries=[mp.Mirror(mp.X, phase=-1)],
    )

    hmv = mp.Harminv(mp.Ex, sources[0].center, fcen, fwidth)
    ldos = mp.Ldos(fcen, fwidth, nfreqs)

    # mmon = sim.add_mode_monitor(
    #     fcen,
    #     fwidth,
    #     nfreqs,
    #     mp.ModeRegion(
    #         center=mp.Vector3(0, -cell.y / 2 + dpml, cell.z / 2),
    #         size=mp.Vector3(cell.x / 2, 0, cell.z),
    #     ),
    # )

    flux_planes = []
    for offset in [-0.5, 0.5]:
        for center, size in [
            [(offset, 0, 0), (0, 1, d_pmma)],
            [(0, offset, 0), (1, 0, d_pmma)],
            [(0, 0, np.sign(offset) * d_pmma / 2), (1, 1, 0)],
        ]:
            flux_planes.append(
                sim.add_flux(
                    fcen,
                    fwidth,
                    nfreqs,
                    mp.FluxRegion(
                        center=cavity_center + mp.Vector3(*center),
                        size=mp.Vector3(*size),
                    ),
                )
            )

    dft_fields = {
        k: sim.add_dft_fields(
            [mp.Ex, mp.Ey, mp.Ez],
            fcen,
            fwidth,
            1,
            center=vol.center,
            size=vol.size,
        )
        for k, vol in [("xy", xy_pmma), ("xy_dr", xy_dr), ("xz", xz), ("xz_wvg", xz_wvg), ("yz", yz)]
    }

    wvg_flux = sim.add_flux(
        fcen,
        fwidth,
        nfreqs,
        mp.FluxRegion(
            center=xz_wvg.center,
            size=mp.Vector3(4 * d_wvg, 0, cell.z),
        ),
    )

    # # fmt: off
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(3, 1)
    # sim.plot2D(output_plane=xy_pmma, ax=ax[0])
    # sim.plot2D(output_plane=xz, ax=ax[1])
    # sim.plot2D(output_plane=yz, ax=ax[2])
    # fig.tight_layout()
    # if mp.am_master():
    #     plt.show()
    # exit()
    # # fmt: on

    sim.run(
        mp.dft_ldos(ldos=ldos),
        mp.after_sources(hmv),
        until_after_sources=mp.stop_when_dft_decayed(decay_by),
    )

    results = {
        "eval_res": sim_res,
        "design": design,
        "index_delta": index_delta,
        "fluxes_design": np.stack([mp.get_fluxes(plane) for plane in flux_planes]),
        "ldos_freqs": np.array(mp.get_ldos_freqs(ldos)),
        "ldos_data_design": np.array(sim.ldos_data),
        "wvg_flux": np.array(mp.get_fluxes(wvg_flux)),
        # "mode_coeffs_design": sim.get_eigenmode_coefficients(mmon, [1]).alpha,
        "harminv": {
            m.freq: {"decay": m.decay, "Q": m.Q, "amp": m.amp, "err": m.err}
            for m in hmv.modes
        },
    }

    for k, dft_mon in dft_fields.items():
        results[f"fields_plane_{k}"] = np.stack(
            [sim.get_dft_array(dft_mon, c, 0) for c in [mp.Ex, mp.Ey, mp.Ez]]
        )

    results["fluxes_bulk"] = bulk_flux_pmma()

    if mp.am_master():
        with h5py.File(f"eval_3d_res{sim_res}_n1019_nomode_dn0130.h5", "w") as f:
            dict_to_h5(results, f)


if __name__ == "__main__":
    main()
