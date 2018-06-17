import os
import numpy as np
import warnings
import gzip

class Eigenval:
    def __init__(self, filename, fermi_level=0):
        if os.path.isfile(filename):
            f = gzip.open(filename, 'rt')
        else:
            f = open(filename.replace('.gz', ''), 'rt')
        header = [f.readline() for _ in range(4)]
        comment = f.readline()
        unknown, npoints, nbands = [int(x) for x in f.readline().split()]
        blankline = f.readline()
        self.nbands = nbands

        self.spin_up = [[] for i in range(nbands)]
        self.spin_down = [[] for i in range(nbands)]

        self.k_points = []
        for i in range(npoints):
            x, y, z, weight = [float(x) for x in f.readline().split()]
            self.k_points.append([x,y,z])

            for j in range(nbands):
                fields = f.readline().split()
                if fields[1] == '************':
                    fields[1] = np.nan
                if fields[2] == '************':
                    fields[2] = np.nan
                id, energy1, energy2 = int(fields[0]), float(fields[1]), float(fields[2])
                self.spin_up[id-1].append(energy1 - fermi_level)
                self.spin_down[id-1].append(energy2 - fermi_level)
            blankline = f.readline()
        f.close()


class Doscar:
    def __init__(self, filename):
        if os.path.isfile(filename):
            f = gzip.open(filename, 'rt')
        else:
            f = open(filename.replace('.gz', ''), 'rt')

        header = [f.readline() for _ in range(4)]
        comment = f.readline()
        line = f.readline()
        if line == '':
            self.converged = False
        else:
            self.converged = True
            self.fermi_energy = float(line.split()[3])


# https://stackoverflow.com/a/312464
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

class Kpoints:
    def __init__(self, filename):
        if os.path.isfile(filename):
            f = gzip.open(filename, 'rt')
        else:
            f = open(filename.replace('.gz', ''), 'rt')

        f.readline()
        self.intersections = int(f.readline())
        if f.readline() != 'Line_mode\n':
            print('Unsupported KPOINTS format')
        f.readline()

        self.k_points = []
        for z in f.readlines():
            z = z.strip()
            if z != "":
                k, name = z.split(" ! ")
                if name == "\Gamma":
                    name = "Γ"

                k = np.array([float(ki) for ki in k.split(" ")])
                self.k_points.append((name, k))
