# Fake data source for testing
import numpy as np
np.random.seed(0)

k = np.linspace(0, 10, 200)
dk = k[1] - k[0]
p = np.poly1d(np.polyfit(k, 1 + 1.0*(np.random.rand(len(k)) - .5), 8))
fake_E_upper = p(k)
p = np.poly1d(np.polyfit(k, -1 + 1.0*(np.random.rand(len(k)) - .5), 8))
fake_E_lower = p(k)

def place_crossing(position, width=.4):
    E = np.interp(np.linspace(0, 1, int(width/dk)), [0, .49, .5, .51, 1], [1, 0, 0, 0, 1])

    for i,j in enumerate(range(position, position+len(E))):
        fake_E_upper[j] = E[i]
        fake_E_lower[j] = -E[i]

def place_parabola(position, width=.4, gap=0):
    k = np.linspace(-1, 1, int(width/dk))
    E = (1-gap)*k**2 + gap

    for i,j in enumerate(range(position, position+len(E))):
        fake_E_upper[j] = E[i]
        fake_E_lower[j] = -E[i]

def place_mexican(position, width=.4):
    k = [0, .25, .5, .75, 1]
    E = [1, 0.4, 1, 0.4, 1]
    p = np.poly1d(np.polyfit(k, E, 4))
    k = np.linspace(0, 1, int(width/dk))

    for i,j in enumerate(range(position, position+len(k))):
        fake_E_upper[j] = p(k[i])
        fake_E_lower[j] = -p(k[i])

place_crossing(19, width=.4)
place_parabola(40, width=.8, gap=.2)
place_parabola(60, width=1.4, gap=.7)
place_crossing(80, width=1.0)
place_parabola(95, width=1.1, gap=.9)
place_parabola(120, width=1.0, gap=0)
place_mexican(150, width=.6)
place_crossing(170, width=0.4)
place_parabola(180, width=.4, gap=.4)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.plot(k, fake_E_upper, 'k')
    plt.plot(k, fake_E_lower, 'k')
    plt.ylabel('Energy')
    plt.xlabel('k')
    plt.savefig('misc/fake_data.png')
    plt.show()
