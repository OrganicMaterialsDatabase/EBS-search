import numpy as np

def path_1D(k):
    """
    Fold out path in 3D along a 1D line by taking the norm between each set of points.
    """
    norm_path = [0]
    for i in range(len(k)-1):
        norm_path.append(np.linalg.norm(k[i+1]-k[i]))
    return np.cumsum(norm_path)

def interpolate(x, line, dimensions):
    """
    Interpolate so that we have the number of dimensions if there are too few points.
    """
    if len(x) != dimensions:
        new_x = np.linspace(np.min(x), np.max(x), dimensions)
        line = np.interp(new_x, x, line)
    return line

def interpolate_normalize(k, E, dimensions):
    """
    First interpolate (see interpolate). Next, subtract the mean and normalize a single band.
    """
    E_interp = interpolate(k, E, dimensions)
    E_interp = E_interp - E_interp.mean()
    if np.linalg.norm(E_interp) > 1e-5:
        E_interp = E_interp / np.linalg.norm(E_interp)
    return E_interp

