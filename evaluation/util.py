import h5py
import numpy as np


def dict_to_h5(d, f):
    for k, v in d.items():
        if isinstance(v, dict):
            g = f.create_group(str(k))
            dict_to_h5(v, g)
        else:
            compression = None if np.isscalar(v) else "gzip"
            f.create_dataset(str(k), data=v, compression=compression)


def h5_to_dict(f, d={}):
    for k, v in f.items():
        if isinstance(v, h5py.Group):
            d[k] = {}
            h5_to_dict(v, d[k])
        elif isinstance(v, h5py.Dataset):
            d[k] = v[()]
    return d
