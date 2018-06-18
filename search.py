import argparse
from annoy import AnnoyIndex
import numpy as np
from lib import vasp
from lib.preprocessing import interpolate_normalize, path_1D
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dimensions', type=int, default=16, help='number of data points for each band')
parser.add_argument('--band_index', type=int, default=0, help='band index relative to Fermi level')
parser.add_argument('--width', type=float, default=.4, help='sliding window width')
opt = parser.parse_args()
print(opt)

annoyindex = AnnoyIndex(int(2*opt.dimensions), metric='angular')
annoyindex.load('index_%d.ann' % opt.band_index)

lookuptable = np.load('lookuptable_%d.npy' % opt.band_index)

folder = str(int(lookuptable[0][0]))

def plot_band_structure(folder, k_highlight):
    kpoints = vasp.Kpoints('data/' + folder + '/KPOINTS.gz')
    doscar = vasp.Doscar('data/' + folder + '/DOSCAR.gz')
    fermi_energy = doscar.fermi_energy
    eigenval = vasp.Eigenval('data/' + folder + '/EIGENVAL.gz', fermi_level=fermi_energy)

    bands = eigenval.spin_up
    lower_fermi_band_index = np.argmax(np.nanmax(bands,axis=1) > 0) - 1

    segment_sizes = kpoints.segment_sizes
    segmented_lower_band = np.split(bands[lower_fermi_band_index+opt.band_index], np.cumsum(segment_sizes))[:-1]
    segmented_upper_band = np.split(bands[lower_fermi_band_index+opt.band_index+1], np.cumsum(segment_sizes))[:-1]
    segmented_kpoints = np.split(eigenval.k_points, np.cumsum(segment_sizes))[:-1]

    last_k = 0
    for k, band_l, band_u in zip(segmented_kpoints, segmented_lower_band, segmented_upper_band):
        k_1D = np.array(path_1D(k)) + last_k
        plt.plot(k_1D, band_l, 'k')
        plt.plot(k_1D, band_u, 'k')
        last_k = np.max(k_1D)

    plt.axvspan(k_highlight, k_highlight+opt.width, color='red', alpha=0.5)

    # Collect axis ticks (high symmetry points) from KPOINTS.gz
    last_k = 0
    label_k = []
    label_names = []
    for left_k, right_k in vasp.chunks(kpoints.k_points, 2):
        if len(label_names) == 0:
            label_k.append(last_k)
            label_names.append(left_k[0])
        elif label_names[-1] != left_k[0]:
            label_names[-1] += (';' + left_k[0])
        last_k += np.linalg.norm(right_k[1] - left_k[1])
        label_k.append(last_k)
        label_names.append(right_k[0])
    plt.xticks(label_k, label_names)

    plt.ylabel('Energy')
    plt.xlabel('k')
    plt.grid(True)
    plt.savefig('search_result.png')
    plt.show()


# Perform the search
search_upper = interpolate_normalize([0, .5, 1], [1, 0, 1], opt.dimensions)
search_lower = interpolate_normalize([0, .5, 1], [-1, 0, -1], opt.dimensions)
search_vector = np.concatenate([search_lower, search_upper])
results = annoyindex.get_nns_by_vector(search_vector, 1, search_k=-1, include_distances=True)
for result, distance in zip(*results):
    folder, k, gap = lookuptable[result]
    print('Angular distance =', distance, 'k =', k)
    plot_band_structure(str(int(folder)), k)
