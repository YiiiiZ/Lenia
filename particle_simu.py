import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation
import IPython.display
from particle_lenia_main import Particle_Lenia
from tools import particle_visualisation


#animation
mu_k = 4
sigma_k = 1
w_k = 0.022
mu_g = 0.6
sigma_g = 0.15
c_rep = 1.0
dt = 0.1
E = []#store energy at time t for following plot
no_frames = 3000

p0 = np.random.uniform(low=-6, high=6, size=(200, 2))
particle_sys = Particle_Lenia(mu_k, sigma_k, w_k, mu_g, sigma_g, c_rep, dt)

fig, ax, img = particle_visualisation(p0)
ani = animation.FuncAnimation(fig, particle_sys.update_particle, fargs=(img, p0), frames=no_frames, interval=5)
ani.save('particle.gif', writer='pillow', fps=20)


#energy analysis
frame_number = np.arange(0, no_frames + 1)
time = dt * frame_number

plt.figure() 
plt.plot(time, E, linestyle='-', color='b')
plt.xlabel('Time (s)')
plt.ylabel('Total Energy (E)')
plt.title('Total Energy against Time')
plt.show()