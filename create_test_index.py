import os
import argparse
from tqdm import tqdm
from lib import vasp
from lib.preprocessing import interpolate, interpolate_normalize
from lib import fake
import numpy as np
from annoy import AnnoyIndex
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
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

k_strided = [fake.k[i] for i in range(0, len(fake.k), opt.stride)]

for window_left in k_strided:
    window_right = window_left + opt.width
    if window_right > np.max(fake.k):
        break
    selection = ((fake.k >= window_left) & (fake.k <= window_right))
    window_k = fake.k[selection]
    window_band_l = fake.E_lower[selection]
    window_band_u = fake.E_upper[selection]
    window_size = np.max(window_k) - np.min(window_k)

    gap = np.min(window_band_u) - np.max(window_band_l)
    window_band_l = interpolate_normalize(window_k, window_band_l, opt.dimensions)
    window_band_u = interpolate_normalize(window_k, window_band_u, opt.dimensions)

    #plt.plot(window_band_l)
    #plt.plot(window_band_u)
    #plt.title('k =' + str(window_left) + ' gap = '+ str( gap))
    #plt.show()

    annoyindex.add_item(len(lookuptable), np.concatenate([window_band_l, window_band_u]))
    lookuptable.append([window_left, gap])

annoyindex.build(opt.trees)
annoyindex.save('index_test.ann')
np.save('lookuptable_test', lookuptable)
