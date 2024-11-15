import numpy as np
import scipy
import jax
import jax.numpy as jp

class Particle_Lenia:
  def __init__(self, mu_k, sigma_k, w_k, mu_g, sigma_g, c_rep, dt):
    #set param
    self.mu_k = mu_k
    self.sigma_k = sigma_k
    self.w_k = w_k
    self.mu_g = mu_g
    self.sigma_g = sigma_g
    self.c_rep = c_rep
    self.dt = dt

  def gaussian_bump(self, r, mu, sigma):
    #for creating U and G functions
    return jp.exp(-((r - mu) / sigma) ** 2)

  def e_field(self, points, x):
    #E=R(r)-G(U), find E field
    r = jp.sqrt(jp.square(x - points).sum(-1).clip(1e-10)) # return the distance array for all pts to x
    U = self.gaussian_bump(r, self.mu_k, self.sigma_k).sum() * self.w_k #Lenia field
    G = self.gaussian_bump(U, self.mu_g, self.sigma_g) #Growth field
    R = self.c_rep * 0.5 * ((1.0 - r).clip(0.0) ** 2).sum() #repulsion field
    E = R - G
    return E

  def grad_e(self, points):
    #find the gradient of E using jax.grad
    gradient = jax.grad(lambda x: self.e_field(points, x))
    return jax.vmap(gradient)(points)#vectorise the result


  def total_energy(self, points):
    # Vectorize the e_field calculation over all points
    energies = jax.vmap(lambda x: self.e_field(points, x))(points)
    # Sum all energies to get total energy
    total_E = energies.sum()
    return total_E

  def update_particle(self, frameNum, img, points):
    #iteration step
    global E
    delta_p = self.dt * (-1) * self.grad_e(points)
    new_points = points + delta_p #since dt is small enough, euler method will do the integration
    img.set_data(new_points[:, 0], new_points[:, 1]) #store x and y coordinate
    points[:] = new_points[:] #update points
    current_e = self.total_energy(points)
    E.append(current_e.item())
    return img,
