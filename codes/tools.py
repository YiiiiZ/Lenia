import matplotlib.pylab as plt
import numpy as np

#random lattice
def random_lattice(N): # Create random lattice with 0 and 1
  return np.random.randint(2, size=(N, N))

#visualisation function
def create_animation(lattice):
  fig, ax = plt.subplots()
  img = ax.matshow(lattice, cmap='GnBu') #discrete lattice w/o interpolation
  plt.colorbar(img, ax=ax)
  plt.close(fig)
  return fig, ax, img

def create_animation_smooth(lattice):
  fig, ax = plt.subplots()
  img = ax.matshow(lattice, cmap='GnBu', interpolation='bilinear') #smooth lattice by using interpolation
  plt.colorbar(img, ax=ax)
  plt.close(fig)
  return fig, ax, img

def kernel_visualisation_dsct(kernel_matrix):
  plt.figure()
  img = plt.imshow(kernel_matrix, cmap='GnBu')  # Store the object returned by imshow
  ax = plt.gca()
  ax.axis('off')
  #plt.savefig('kernel.png', format='png')
  plt.show()

def kernel_visualisation_ctns(kernel_matrix):
  plt.figure()
  plt.imshow(kernel_matrix, cmap='GnBu', interpolation='bilinear')# Apply interpolation to make it smooth
  ax = plt.gca()
  ax.axis('off')
  #plt.savefig('kernel.png', format='png')
  plt.show()

def particle_visualisation(points):
  fig, ax = plt.subplots(figsize=(6, 6))
  x, y = points[:, 0], points[:, 1]
  img, = ax.plot(x, y, 'o', color='blue', alpha=0.5)
  ax.set_xlim([-12, 12])
  ax.set_ylim([-12, 12])
  ax.set_aspect('equal')
  ax.set_axis_off()
  plt.close(fig)
  return fig, ax, img