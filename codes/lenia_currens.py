import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation
import IPython.display
from tools import create_animation_smooth, random_lattice, kernel_visualisation_ctns
from creatures import acurrens64, acurrens_param
from lenia_main import Lenia, growth


#animation
currens = Lenia() #create instance
currens.set_params(**acurrens_param)  # Unpack the dictionary
A = np.array(acurrens64).copy()


fig, ax, img = create_animation_smooth(A)
ani = animation.FuncAnimation(fig, currens.iteration, fargs=(img, A), frames=300, interval=20, blit=True)
ani.save('currens.gif', writer='pillow', fps=20)


#growth function and kernel visualisation
mu = 0.2
sigma = 0.022
U_values = np.linspace(0, 1, 500)
growth_values = [growth(U, mu, sigma) for U in U_values]

plt.plot(U_values, growth_values, label="Astrium currens\n$\\mu$=0.2, $\\sigma$=0.022")
plt.xlabel('Potential distribution U')
plt.ylabel('Growth function G')
plt.legend()
# plt.show()

kernel_visualisation_ctns(currens.sub_kernel())