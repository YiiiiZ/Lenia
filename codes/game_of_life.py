import numpy as np
import scipy
import matplotlib.pylab as plt
import matplotlib.animation as animation
import IPython.display
from tools import create_animation, random_lattice, kernel_visualisation_dsct

def growth(U):
  return 0 + (U==3) - ((U<2)|(U>3)) # step function equals to 0 at 2, 1 at 3 and -1 otherwise

def iteration(frameNum, img, lattice):
    global K
    new_lattice = lattice.copy()
    U = scipy.signal.convolve2d(lattice, K, mode='same', boundary='wrap')  # potential distribution
    new_lattice = np.clip(lattice + growth(U), 0, 1) # limit the lattice value with between 0 and 1
    img.set_data(new_lattice)
    lattice[:] = new_lattice[:]
    return  (img,)


#animation
K = np.asarray([[1,1,1], [1,0,1], [1,1,1]]) # kernel as 3*3 matrix
N = 64
A = random_lattice(N)

fig, ax, img = create_animation(A)
ani = animation.FuncAnimation(fig, iteration, fargs=(img, A), frames=100, interval=50)
ani.save('GoL.gif', writer='pillow', fps=20)


#growth function and kernel visualisation
neighbor_counts = range(9)
G_values = [growth(n) for n in neighbor_counts]
plt.plot(neighbor_counts, G_values, marker='o', label='Game of Life')
plt.xlabel('Number of Living Neighbors n')
plt.ylabel('Growth Function G')
plt.xticks(neighbor_counts)
plt.legend()
# plt.savefig('game_of_life_step_function.png', format='png', dpi=300)

kernel_visualisation_dsct(K)