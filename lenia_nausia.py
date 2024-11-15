import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation
import IPython.display
from tools import create_animation_smooth, random_lattice, kernel_visualisation_ctns
from creatures import anausia64, anausia_param
from lenia_main import Lenia, growth


#animation
nausia = Lenia() #create instance
nausia.set_params(**anausia_param)  # Unpack the dictionary
A = np.array(anausia64).copy()

fig, ax, img = create_animation_smooth(A)
ani = animation.FuncAnimation(fig, nausia.iteration, fargs=(img, A), frames=100, interval=20, blit=True)
ani.save('nausia.gif', writer='pillow', fps=20)


#growth function and kernel visualisation

mu = 0.22
sigma = 0.021
U_values = np.linspace(0, 1, 500)
growth_values = [growth(U, mu, sigma) for U in U_values]

plt.plot(U_values, growth_values, label="Astrium nausia\n$\\mu$=0.22, $\\sigma$=0.021")
plt.xlabel('Potential distribution U')
plt.ylabel('Growth function G')
plt.legend()
plt.show()

kernel_visualisation_ctns(nausia.sub_kernel())