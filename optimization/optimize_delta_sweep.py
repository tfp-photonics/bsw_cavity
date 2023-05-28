#!/usr/bin/env python3

import datetime
from itertools import count
from pprint import pprint

import meep as mp
import autograd.numpy as anp
import h5py
import meep.adjoint as mpa
import nlopt
import numpy as np
import scipy.ndimage
from meep import timing_measurements as timing
from autograd import tensor_jacobian_product, value_and_grad
from autograd.extend import defvjp, primitive
from skimage.draw import disk

gaussian_filter = primitive(scipy.ndimage.gaussian_filter)
defvjp(
    gaussian_filter,
    lambda ans, x, *args, **kwargs: lambda g: gaussian_filter(g, *args, **kwargs),
)


def projection(x, a, b=0.5):
    num = np.tanh(a * b) + anp.tanh(a * (x - b))
    denom = np.tanh(a * b) + np.tanh(a * (1 - b))
    return num / denom


def gray_indicator(x):
    return anp.mean(4 * x * (1 - x))


def get_parametrization(shape, radius, sigma=1.0, a=20, b=0.5, circle_mask=True):
    shape = np.asarray(shape)
    ks = sigma * shape / radius / np.sqrt(3)

    if circle_mask:
        mask = np.zeros(shape)
        rr, cc = disk((shape - 1) / 2, shape[0] / 2, shape=shape)
        mask[rr, cc] = 1

    def parametrization(x):
        x = anp.reshape(x, (shape[0] // 2, shape[1]))
        x = anp.concatenate([x, x[::-1]])
        x = gaussian_filter(x, ks)
        x = projection(x, a, b)
        if circle_mask:
            x = x * mask
        return x

    return parametrization


mp.verbosity(0)
sim_res = 50
radius = 5.7
pshape = (int(2 * sim_res * radius), int(2 * sim_res * radius))
lcen = 0.57
decay_by = 1e-6
# index_low = 1.082
# index_low = 1.038
# index_high = 1.167
index_low = 1.019127761166746
index_high = 1.1486199768435486

d_wvg = 0.3
l_wvg = 1
dpml = 1.0
fcen = 1 / lcen
fwidth = 0.2

feature_size = 0.05

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

# for index_delta in np.linspace(0.01, 0.1, 19):
# for index_delta in np.linspace(0.105, 0.14, 8):
# for index_delta in [0.13]:
for index_delta in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12]:
# for index_delta in [0.11, 0.12, 0.13]:
    print(f"\nStart of run, {index_delta=}\n", flush=True)
    tstart = datetime.datetime.now()

    n1 = mp.Medium(index=index_low)
    n2 = mp.Medium(index=index_low + index_delta)

    mg = mp.MaterialGrid(pshape, n1, n2)

    design_region = mpa.DesignRegion(
        mg,
        volume=mp.Volume(
            center=mp.Vector3(0, cavity_center),
            size=mp.Vector3(2 * radius, 2 * radius),
        ),
    )

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

    # mmon = sim.add_mode_monitor(
    #     fcen,
    #     0,
    #     1,
    #     mp.ModeRegion(
    #         center=mp.Vector3(0, dpml),
    #         size=mp.Vector3(cell.x, 0),
    #     ),
    #     mode=1,
    #     eig_parity=mp.EVEN_Z,  # TM
    # )
    # sim.run(until=100)
    # sim.get_eigenmode_coefficients(mmon, [1], eig_parity=mp.EVEN_Z)
    # timing_measurements = timing.MeepTimingMeasurements.new_from_simulation(sim)
    # pprint(timing_measurements.measurements)
    # exit()

    te0 = mpa.EigenmodeCoefficient(
        sim,
        mp.Volume(
            center=mp.Vector3(0, dpml),
            size=mp.Vector3(cell.x, 0),
        ),
        mode=1,
        eig_parity=mp.EVEN_Z,  # TM
    )

    mpa_opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=lambda x: anp.mean(anp.abs(x) ** 2),
        objective_arguments=[te0],
        design_regions=[design_region],
        frequencies=[fcen],
        decay_by=decay_by,
    )

    obj_vals = []
    counter = count()
    cst_bnd = 1
    bfac = 20
    x0 = np.full(np.prod(pshape) // 2, 0.5)
    parametrization = get_parametrization(pshape, radius, feature_size, bfac)

    while np.log10(gray_indicator(parametrization(x0))) > -2:
        print(f"\n{bfac=}, {cst_bnd=}\n", flush=True)

        def nlopt_obj(x, gd):
            xp = parametrization(x)
            v, g = mpa_opt([xp.ravel()])
            g = np.reshape(g, pshape)
            if gd.size > 0:
                gd[:] = tensor_jacobian_product(parametrization, 0)(x, g)

            print(f"{next(counter):>4}: {v=}", flush=True)

            obj_vals.append(v)

            return v

        def cst_fun(x, gd):
            xp = parametrization(x)
            v, g = value_and_grad(gray_indicator)(xp)
            if gd.size > 0:
                gd[:] = tensor_jacobian_product(parametrization, 0)(x, g)
            return v - cst_bnd

        opt = nlopt.opt(nlopt.LD_MMA, x0.size)
        opt.set_max_objective(nlopt_obj)
        opt.set_lower_bounds(0)
        opt.set_upper_bounds(1)
        opt.add_inequality_constraint(cst_fun)
        opt.set_ftol_rel(1e-5)

        x0 = opt.optimize(x0)

        bfac *= 2
        cst_bnd = gray_indicator(parametrization(x0))
        parametrization = get_parametrization(pshape, radius, feature_size, bfac)

    if mp.am_master():
        with h5py.File("index_delta_sweep_n1019.h5", "a") as f:
            parametrization = get_parametrization(
                pshape, radius, feature_size, bfac / 2
            )
            grp = f.create_group(f"{index_delta:.3f}")
            grp.attrs["index_delta"] = index_delta
            grp.attrs["tstart"] = str(tstart)
            grp.attrs["tend"] = str(datetime.datetime.now())
            grp.attrs["n1"] = index_low
            grp.attrs["n2"] = index_low + index_delta
            grp.attrs["sim_res"] = sim_res
            grp.attrs["bfac"] = bfac / 2
            grp.attrs["radius"] = radius
            grp.attrs["feature_size"] = feature_size
            grp.create_dataset("x0", data=x0)
            grp.create_dataset("design", data=parametrization(x0))
            grp.create_dataset("hist", data=np.array(obj_vals))
