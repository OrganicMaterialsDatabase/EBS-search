import os
import argparse
from tqdm import tqdm
from lib import vasp
from lib.preprocessing import path_1D, interpolate, interpolate_normalize
import numpy as np
from annoy import AnnoyIndex


parser = argparse.ArgumentParser()
parser.add_argument('--band_index', type=int, default=0, help='band index relative to Fermi level')
parser.add_argument('--width', type=float, default=.4, help='sliding window width')
parser.add_argument('--dimensions', type=int, default=16, help='number of data points for each band')
parser.add_argument('--stride', type=int, default=1, help='number of data points to slide each window (1 means zero datapoints are skipped)')
parser.add_argument('--trees', type=int, default=10, help='number of trees in annoy index')
opt = parser.parse_args()
print(opt)


# Annoy index stores the electronic band structure with each vector corresponding to a sliding window.
annoyindex = AnnoyIndex(int(2*opt.dimensions), metric='angular')

# A lookuptable is necessary to link annoy vectors to their correct material and k-space position.
lookuptable = []


for folder in tqdm(os.listdir('data')):
    kpoints = vasp.Kpoints('data/' + folder + '/KPOINTS.gz')
    doscar = vasp.Doscar('data/' + folder + '/DOSCAR.gz')
    if doscar.converged == False:
        continue # skip calculations that did not converge
    fermi_energy = doscar.fermi_energy
    eigenval = vasp.Eigenval('data/' + folder + '/EIGENVAL.gz', fermi_level=fermi_energy)

    bands = eigenval.spin_up
    # lower_fermi_band_index indicates the band just below Fermi level
    lower_fermi_band_index = np.argmax(np.nanmax(bands,axis=1) > 0) - 1

    # Find continuous segments in KPOINTS
    # For example: segment_sizes => [40, 40, 40, 20]
    segment_sizes = kpoints.segment_sizes

    # Extract the bands and split them in continuous segments
    segmented_lower_band = np.split(bands[lower_fermi_band_index+opt.band_index], np.cumsum(segment_sizes))[:-1]
    segmented_upper_band = np.split(bands[lower_fermi_band_index+opt.band_index+1], np.cumsum(segment_sizes))[:-1]
    segmented_kpoints = np.split(eigenval.k_points, np.cumsum(segment_sizes))[:-1]

    # Process each pair of bands in a continuous segment
    k_norm = 0
    for k, band_l, band_u in zip(segmented_kpoints, segmented_lower_band, segmented_upper_band):
        k_1D = np.array(path_1D(k))
        k_1D_strided = [k_1D[i] for i in range(0, len(k_1D), opt.stride)]

        for window_left in k_1D_strided:
            window_right = window_left + opt.width
            if window_right > np.max(k_1D):
                break
            selection = ((k_1D >= window_left) & (k_1D <= window_right))
            window_k = k_1D[selection]
            window_band_l = band_l[selection]
            window_band_u = band_u[selection]
            window_size = np.max(window_k) - np.min(window_k)

            gap = np.min(window_band_u) - np.max(window_band_l)
            window_band_l = interpolate_normalize(window_k, window_band_l, opt.dimensions)
            window_band_u = interpolate_normalize(window_k, window_band_u, opt.dimensions)
            annoyindex.add_item(len(lookuptable), np.concatenate([window_band_l, window_band_u]))
            lookuptable.append([int(folder), k_norm + window_left, gap])

        k_norm += np.max(k_1D)
        
annoyindex.build(opt.trees)
annoyindex.save('index_%d.ann' % opt.band_index)
np.save('lookuptable_%d' % opt.band_index, lookuptable)
print('%d vectors stored in index' % len(lookuptable))
