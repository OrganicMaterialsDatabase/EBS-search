import argparse
from annoy import AnnoyIndex
import numpy as np
from lib import vasp
from lib.preprocessing import interpolate_normalize, path_1D, plot_band_structure

parser = argparse.ArgumentParser()
parser.add_argument('--dimensions', type=int, default=16, help='number of data points for each band')
parser.add_argument('--band_index', type=int, default=0, help='band index relative to Fermi level')
parser.add_argument('--width', type=float, default=.4, help='sliding window width')
parser.add_argument('--pattern', required=True, help='crossing | parabola | mexican')
parser.add_argument('--search_k', type=int, default=-1, help='ANNOY search_k parameter')
opt = parser.parse_args()
print(opt)

annoyindex = AnnoyIndex(int(2*opt.dimensions), metric='angular')
annoyindex.load('index_%d.ann' % opt.band_index)

lookuptable = np.load('lookuptable_%d.npy' % opt.band_index)

# Define a search pattern
if opt.pattern == 'crossing':
    search_upper = interpolate_normalize([0, .5, 1], [1, 0, 1], opt.dimensions)
    search_lower = interpolate_normalize([0, .5, 1], [-1, 0, -1], opt.dimensions)
    search_vector = np.concatenate([search_lower, search_upper])
elif opt.pattern == 'parabola':
    k = np.linspace(-1,1, 100)
    search_upper = interpolate_normalize(k, k**2, opt.dimensions)
    search_lower = interpolate_normalize(k, -1*(k**2), opt.dimensions)
    search_vector = np.concatenate([search_lower, search_upper])
elif opt.pattern == 'mexican':
    k = [0, .15, .28, .5, .72, .85, 1]
    E = [1, .9, 0, 1, 0, .9, 1]
    p = np.poly1d(np.polyfit(k, E, 6))
    k = np.linspace(.12, .88, 1000)
    search_upper = interpolate_normalize(k, [p(x) for x in k], opt.dimensions)
    search_lower = interpolate_normalize(k, [-p(x) for x in k], opt.dimensions)
    search_vector = np.concatenate([search_lower, search_upper])
else:
    raise ValueError('--pattern argument unrecognized')


# Plot search pattern
#import matplotlib.pyplot as plt
#plt.plot(search_upper, 'r')
#plt.plot(search_lower, 'b')
#plt.show()


# Search
results = annoyindex.get_nns_by_vector(search_vector, 1, search_k=opt.search_k, include_distances=True)
for result, distance in zip(*results):
    folder, k, gap = lookuptable[result]
    print('COD =', int(folder), 'Angular distance =', distance, 'k =', k)
    plot_band_structure(str(int(folder)), opt.band_index, opt.width, k, opt.pattern)
