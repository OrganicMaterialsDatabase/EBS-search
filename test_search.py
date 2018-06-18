import argparse
from annoy import AnnoyIndex
import numpy as np
from lib import vasp
from lib.preprocessing import interpolate_normalize, path_1D, plot_band_structure

from lib import fake
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dimensions', type=int, default=16, help='number of data points for each band')
parser.add_argument('--results', type=int, default=5, help='# of results')
parser.add_argument('--band_index', type=int, default=0, help='band index relative to Fermi level')
parser.add_argument('--width', type=float, default=.4, help='sliding window width')
parser.add_argument('--pattern', required=True, help='crossing | parabola | mexican')
opt = parser.parse_args()
print(opt)

annoyindex = AnnoyIndex(int(2*opt.dimensions), metric='angular')
annoyindex.load('index_test.ann')

lookuptable = np.load('lookuptable_test.npy')

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
    k = [0, .25, .5, .75, 1]
    E = [1, 0, 1, 0, 1]
    p = np.poly1d(np.polyfit(k, E, 4))
    k = np.linspace(0, 1, 1000)
    search_upper = interpolate_normalize(k, [p(x) for x in k], opt.dimensions)
    search_lower = interpolate_normalize(k, [-p(x) for x in k], opt.dimensions)
    search_vector = np.concatenate([search_lower, search_upper])
else:
    raise ValueError('--pattern argument unrecognized')


# Plot search pattern
#plt.plot(search_upper)
#plt.plot(search_lower)
#plt.show()

plt.plot(fake.k, fake.E_upper, 'k')
plt.plot(fake.k, fake.E_lower, 'k')

# Search
results = annoyindex.get_nns_by_vector(search_vector, opt.results, search_k=-1, include_distances=True)
for result, distance in zip(*results):
    k, gap = lookuptable[result]
    print('Angular distance =', distance, 'k =', k)
    plt.axvspan(k, k+opt.width, color='red', alpha=0.5)

plt.ylabel('Energy')
plt.xlabel('k')
plt.savefig('misc/test_results_'+opt.pattern+'.png')
plt.show()
