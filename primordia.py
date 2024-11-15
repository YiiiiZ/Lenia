import numpy as np
import scipy
import matplotlib.pylab as plt
import matplotlib.animation as animation
import IPython.display
from tools import create_animation, random_lattice, kernel_visualisation_dsct


#update freq T = 1/dt
def ctns_state_lattice(N):
  return np.random.rand(N, N) #create N*N random number from [0, 1]

def growth(U): #update to continuous growth function
  global K, state
  return (0 + ((U>=0.25)&(U<=0.35)) - ((U<=0.20)|(U>=0.30)))

def iteration(frameNum, img, lattice):
    global K, T, state
    new_lattice = lattice.copy()
    U = scipy.signal.convolve2d(lattice, K, mode='same', boundary='wrap') # normalised potential distribution
    new_lattice = np.clip(lattice + (1/T) * growth(U), 0, 1) # limit the lattice value with between 0 and 1
    img.set_data(new_lattice)
    lattice[:] = new_lattice[:]
    return  (img,)


#animation
T = 10
N = 64
A = ctns_state_lattice(N)
K = np.asarray([[1,1,1], [1,0,1], [1,1,1]])/ 8

fig, ax, img = create_animation(A)
ani = animation.FuncAnimation(fig, iteration, fargs=(img, A), frames=100, interval=50, blit=True)
ani.save('primordia.gif', writer='pillow', fps=20)


#growth function and kernel visualisation
U = np.linspace(0, 1, 500)
G_values = np.where((U >= 0.25) & (U <= 0.35), 1, np.where((U <= 0.20) | (U >= 0.30), -1, 0))
plt.plot(U, G_values, label="Primordia")
plt.xlabel('Number of Living Neighbors n')
plt.ylabel('Growth Function G')
plt.yticks([-1, 0, 1])
plt.legend()
#plt.savefig('Pmd_gf.png', format='png', dpi=300)
plt.show()

kernel_visualisation_dsct(K)