# Fake data source for testing
import numpy as np
np.random.seed(0)

k = np.linspace(0, 10, 1000)
dk = k[1] - k[0]
p = np.poly1d(np.polyfit(k, 1 + 1.0*(np.random.rand(len(k)) - .5), 8))
E_upper = p(k)
p = np.poly1d(np.polyfit(k, -1 + 1.0*(np.random.rand(len(k)) - .5), 8))
E_lower = p(k)

def place_crossing(position, width=.4):
    E = np.interp(np.linspace(0, 1, int(width/dk)), [0, .49, .5, .51, 1], [1, 0, 0, 0, 1])

    for i,j in enumerate(range(position, position+len(E))):
        E_upper[j] = E[i]
        E_lower[j] = -E[i]

def place_parabola(position, width=.4, gap=0):
    k = np.linspace(-1, 1, int(width/dk))
    E = (1-gap)*k**2 + gap

    for i,j in enumerate(range(position, position+len(E))):
        E_upper[j] = E[i]
        E_lower[j] = -E[i]

def place_mexican(position, width=.4):
    k = [0, .25, .5, .75, 1]
    E = [1, .4, 1, .4, 1]
    p = np.poly1d(np.polyfit(k, E, 4))
    k = np.linspace(0, 1, int(width/dk))

    for i,j in enumerate(range(position, position+len(k))):
        E_upper[j] = p(k[i])
        E_lower[j] = -p(k[i])

place_crossing(int(19/200*len(k)), width=.4)
place_parabola(int(40/200*len(k)), width=.8, gap=.2)
place_parabola(int(60/200*len(k)), width=1.4, gap=.7)
place_crossing(int(80/200*len(k)), width=1.0)
place_parabola(int(95/200*len(k)), width=1.1, gap=.9)
place_parabola(int(120/200*len(k)), width=0.4, gap=0)
place_mexican(int(150/200*len(k)), width=.4)
place_crossing(int(170/200*len(k)), width=0.4)
place_parabola(int(180/200*len(k)), width=.4, gap=.4)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,3))
    plt.plot(k, E_upper, 'k')
    plt.plot(k, E_lower, 'k')
    plt.ylabel('Energy')
    plt.xlabel('k')
    plt.savefig('misc/fake_data.png')
    plt.show()
