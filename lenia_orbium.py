import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation
import IPython.display
from tools import create_animation_smooth, random_lattice, kernel_visualisation_ctns
from creatures import two_Orbium64, anausia_param
from lenia_main import Lenia, growth, kernel, iteration


# 64 * 64, two orbium animation
R = 13
T = 10
N = 64
mu = 0.15
sigma = 0.017
K = kernel(R, N)
A = np.array(two_Orbium64).copy()

fig, ax, img = create_animation_smooth(A)
ani = animation.FuncAnimation(fig, iteration, fargs=(img, A, K, T, mu, sigma), frames=100, interval=50, blit=True)
ani.save('orbium.gif', writer='pillow', fps=20)
