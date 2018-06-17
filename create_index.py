import os
import argparse
from tqdm import tqdm
from lib import vasp
import numpy as np
from annoy import AnnoyIndex


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='folder | fake')
parser.add_argument('--band_index', type=int, default=0, help='band index relative to Fermi level')
parser.add_argument('--width', type=float, default=.4, help='sliding window width')
parser.add_argument('--dimensions', type=int, default=16, help='number of data points for each band')
parser.add_argument('--stride', type=int, default=0, help='number of data points to skip')
parser.add_argument('--trees', type=int, default=10, help='number of trees in annoy index')
opt = parser.parse_args()
print(opt)


annoyindex = AnnoyIndex(int(2*opt.dimensions))
lookuptable = []

def path_1D(k):
    """
    Fold out path in 3D along a 1D line by taking the norm between each set of points.
    """
    norm_path = [0]
    for i in range(len(k)-1):
        norm_path.append(np.linalg.norm(k[i+1]-k[i]))
    return np.cumsum(norm_path)

def interpolate(x, line):
    """
    Interpolate so that we have the number of dimensions if there are too few points.
    """
    if len(x) != opt.dimensions:
        new_x = np.linspace(np.min(x), np.max(x), opt.dimensions)
        line = np.interp(new_x, x, line)
    return line

def interpolate_normalize(k, E):
    """
    First interpolate (see interpolate). Next, subtract the mean and normalize a single band.
    """
    E_interp = interpolate(k, E)
    E_interp = E_interp - E_interp.mean()
    if np.linalg.norm(E_interp) > 1e-5:
        E_interp = E_interp / np.linalg.norm(E_interp)
    return E_interp


for folder in tqdm(os.listdir('data')):
    kpoints = vasp.Kpoints('data/' + folder + '/KPOINTS.gz')
    doscar = vasp.Doscar('data/' + folder + '/DOSCAR.gz')
    if doscar.converged == False:
        continue # skip calculations that did not converge
    fermi_energy = doscar.fermi_energy
    eigenval = vasp.Eigenval('data/' + folder + '/EIGENVAL.gz', fermi_level=fermi_energy)

    bands = eigenval.spin_up
    lower_fermi_band_index = np.argmax(np.nanmax(bands,axis=1) > 0)

    # Find continuous segments in KPOINTS
    # For example: segment_sizes => [40, 40, 40, 20]
    last_k = None
    segment_sizes = []
    for segment in vasp.chunks(kpoints.k_points, 2):
        # segment => [('X', array([0.5, 0. , 0. ])), ('Γ', array([0., 0., 0.]))]
        if np.array_equal(segment[0][1], last_k):
            segment_sizes[-1] += kpoints.intersections
        else:
            segment_sizes.append(kpoints.intersections)
        last_k = segment[1][1]

    # Extract the bands and split them in continuous segments
    segmented_lower_band = np.split(bands[lower_fermi_band_index+opt.band_index], np.cumsum(segment_sizes))[:-1]
    segmented_upper_band = np.split(bands[lower_fermi_band_index+opt.band_index+1], np.cumsum(segment_sizes))[:-1]
    segmented_kpoints = np.split(eigenval.k_points, np.cumsum(segment_sizes))[:-1]

    # Process each pair of bands in a continuous segment
    k_norm = 0
    for k, band_l, band_u in zip(segmented_kpoints, segmented_lower_band, segmented_upper_band):
        k_1D = np.array(path_1D(k))

        for window_left in k_1D:
            window_right = window_left + opt.width
            if window_right > np.max(k_1D):
                break
            selection = ((k_1D >= window_left) & (k_1D <= window_right))
            window_k = k_1D[selection]
            window_band_l = band_l[selection]
            window_band_u = band_u[selection]
            window_size = np.max(window_k) - np.min(window_k)

            gap = np.min(window_band_u) - np.max(window_band_l)
            window_band_l = interpolate_normalize(window_k, window_band_l)
            window_band_u = interpolate_normalize(window_k, window_band_u)
            annoyindex.add_item(len(lookuptable), np.concatenate([window_band_l, window_band_u]))
            lookuptable.append([int(folder), k_norm + window_left, gap])

        k_norm += np.max(k_1D)
        
annoyindex.build(opt.trees)
annoyindex.save('index_%d.ann' % opt.band_index)
np.save('lookuptable', lookuptable)
