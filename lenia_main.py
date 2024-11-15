import numpy as np
import scipy


class Lenia:

  def __init__(self, R=13, peaks=None, mu=0.15, sigma=0.017, dt=0.1, N = 64, kernel_type=0, delta_type=0):
    #set the default param
    self.R = R
    self.peaks = peaks if peaks is not None else [1]
    self.mu = mu
    self.sigma = sigma
    self.dt = dt
    self.N = N
    self.kernel_type = kernel_type
    self.delta_type = delta_type


  def set_params(self, R, peaks, mu, sigma, dt, N, kernel_type=0, delta_type=0):
    #store param if provided
    self.R = R
    self.peaks = peaks
    self.mu = mu
    self.sigma = sigma
    self.dt = dt
    self.kernel_type = kernel_type
    self.delta_type = delta_type

  def kernel_core(self, r):
    #kernel core function
    if self.kernel_type == 0:
        return (4 * r * (1 - r)) ** 4
    else:
        return np.exp(4 - 1 / (r * (1 - r)))

  def kernel_shell(self, distance):
    #kernel layer function
    n = len(self.peaks) # n = no. rings in the kernel function
    nd = n * distance
    entry = nd % 1 # take decimal values
    peak_index = np.minimum(np.floor(nd).astype(int), n-1) #this enables difft index to be chosen for difft nr
    peak = np.array(self.peaks)[peak_index] #pick the corresponding peak index
    return self.kernel_core(entry) * peak #return the combined kernel elemenet matrix

  def kernel(self, R):
    x, y = np.ogrid[-self.N/2:self.N/2, -self.N/2:self.N/2] #create empty array with same size as the lattice
    distance_grid = np.sqrt(x**2 + y**2) #calculate distance
    distance_array = np.where(distance_grid > R, 0, distance_grid)
    normalised_distance = distance_array / self.R
    Kernel = self.kernel_shell(normalised_distance)
    return Kernel/np.sum(Kernel)

  def growth_function(self, U):
    return 2 * np.exp(-0.5 * ((U - self.mu) / self.sigma) ** 2) - 1

  def iteration(self, frameNum, img, lattice):
    #updated
    new_lattice = lattice.copy()
    ft_K = np.fft.fft2(np.fft.fftshift(self.kernel(self.R)))
    U = np.real(np.fft.ifft2(ft_K * np.fft.fft2(new_lattice)))
    new_lattice = np.clip(new_lattice + self.dt * self.growth_function(U), 0, 1)
    img.set_data(new_lattice)
    lattice[:] = new_lattice[:]
    return (img,)
  
  def sub_kernel(self): #To focus on the central part of the kernel
    orginal_k = self.kernel(self.R)
    N1 = int((self.N - 2 * self.R) / 2)
    N2 = int((self.N - 2 * self.R)/ 2 + 2 * self.R)
    sub_K = orginal_k[N1-4: N2+5, N1-4:N2+5] #modify the position to make it central and leave some space
    return sub_K


#Alternatively, Lenia for simply kernel creatures can be implemented w/o a class.


def kernel_function(x):
  global R
  return np.exp(4-1/((1/R) * x - ((1/R) * x)**2))

def kernel(R, N): #update to the right dimension
  kernel_function = lambda x: np.exp(4-1/((1/R) * x - ((1/R) * x)**2))
  x, y = np.ogrid[-N/2:N/2, -N/2:N/2] # to make K same dimension as lattice to perform ift later on
  distance_grid = np.sqrt(x**2 + y**2)
  distance_grid[distance_grid > R] = 0
  Kernel = kernel_function(distance_grid)
  return Kernel/np.sum(Kernel)

def growth(U, mu, sigma): #keep the same
  return 2 * np.exp(-0.5 * ((U -mu)/sigma) ** 2) - 1

def iteration(frameNum, img, lattice, K, T, mu, sigma): #modify iteration function using FFT
    new_lattice = lattice.copy()
    ft_K = np.fft.fft2(np.fft.fftshift(K)) #fourier transform of kernel K
    U = np.real(np.fft.ifft2(ft_K * np.fft.fft2(lattice))) #replace potential distribution to inverse FT of a product of two FTs
    new_lattice = np.clip(lattice + (1/T) * growth(U, mu, sigma), 0, 1)
    img.set_data(new_lattice)
    lattice[:] = new_lattice[:]
    return  (img,)

