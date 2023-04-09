
#import lib

#print(lib.Monte_Carlo_pi(10))

#https://github.com/louishrm/Quantum-Tunneling/blob/main/QM%20Tunnelling.ipynb

import numpy as np
x = np.linspace(-10,10,5000)
deltax = x[1]-x[0]

def norm(phi):
    norm = np.sum(np.square(np.abs(phi)))*deltax
    return phi/np.sqrt(norm)

def complex_plot(x,y,prob=True,**kwargs):
    real = np.real(y)
    imag = np.imag(y)
    a,*_ = plt.plot(x,real,label='Re',**kwargs)
    b,*_ = plt.plot(x,imag,label='Im',**kwargs)
    plt.xlim(-2,2)
    if prob:
        p,*_ = plt.plot(x,np.abs(y),label='$\sqrt{P}$')
        return a,b,p
    else:
        return a,b
    
def wave_packet(pos=0,mom=0,sigma=0.2):
    return norm(np.exp(-1j*mom*x)*np.exp(-np.square(x-pos)/sigma/sigma,dtype=complex))
                
def d_dxdx(phi,x=x):
    dphi_dxdx = -2*phi
    dphi_dxdx[:-1] += phi[1:]
    dphi_dxdx[1:] += phi[:-1]
    return dphi_dxdx/deltax

def d_dt(phi,h=1,m=100,V=0):
    return 1j*h/2/m * d_dxdx(phi) - 1j*V*phi/h

def euler(phi, dt, **kwargs):
    return phi + dt * d_dt(phi, **kwargs)

def rk4(phi, dt, **kwargs):
    k1 = d_dt(phi, **kwargs)
    k2 = d_dt(phi+dt/2*k1, **kwargs)
    k3 = d_dt(phi+dt/2*k2, **kwargs)
    k4 = d_dt(phi+dt*k3, **kwargs)
    return phi + dt/6*(k1+2*k2+2*k3+k4)

def simulate(phi_sim, 
             method='rk4', 
             V=0, 
             steps=100000, 
             dt=1e-1, 
             condition=None, 
             normalize=True,
             save_every=100):
    simulation_steps = [np.copy(phi_sim)]
    for i in range(steps):
        if method == 'euler':
            phi_sim = euler(phi_sim,dt,V=V)
        elif method == 'rk4':
            phi_sim = rk4(phi_sim,dt,V=V)
        else:
            raise Exception(f'Unknown method {method}')
        if condition:
            phi_sim = condition(phi_sim)
        if normalize:
            phi_sim = norm(phi_sim)
        if save_every is not None and (i+1) % save_every == 0:
            simulation_steps.append(np.copy(phi_sim))
    return simulation_steps

sim_free = simulate(wave_packet(mom=-10),steps=200000,save_every=1000)#wave_packet()

from matplotlib.animation import FuncAnimation

import matplotlib.pyplot as plt

def animate(simulation_steps,init_func=None):
    fig = plt.figure()
    re,im,prob = complex_plot(x,simulation_steps[0])
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    if init_func:
        init_func()
    plt.legend()

    def animate(frame):
        prob.set_data((x, np.abs(simulation_steps[frame])))
        re.set_data((x, np.real(simulation_steps[frame])))
        im.set_data((x, np.imag(simulation_steps[frame])))
        return prob,re,im

    anim = FuncAnimation(fig, animate, frames=int(len(simulation_steps)), interval=50)
    plt.ioff()
    plt.show()

    return anim

animate(sim_free)

box_potential = np.where((x>-2)&(x<2),0,1)
sim_box_mom = simulate(wave_packet(mom=10),V=box_potential,steps=100000,save_every=500)

def box_init():
    plt.gcf().axes[0].axvspan(2, 3, alpha=0.2, color='red')
    plt.gcf().axes[0].axvspan(-3, -2, alpha=0.2, color='red')
    plt.xlim(-3,3)
    plt.ylim(-3,3)

animate(sim_box_mom,init_func=box_init)

barrier_weak_potential = np.where((x>1.4)&(x<1.6),3.5e-2,0)
sim_barrier_mom = simulate(wave_packet(mom=-10),V=barrier_weak_potential,steps=50000,save_every=500)

def barrier_init():
    plt.gcf().axes[0].axvspan(1.4, 1.6, alpha=0.2, color='orange')
    plt.xlim(-2,4)
    plt.ylim(-3,3)

animate(sim_barrier_mom,init_func=barrier_init)

quadratic_potential = 1e-2*np.square(x)
sim_quadratic_potential = simulate(wave_packet(mom=-10),V=quadratic_potential,steps=400000,save_every=500)

def quadratic_init():
    plt.fill_between(x,(np.square(x)-3),-3,color='orange',alpha=0.2)
    plt.xlim(-3,3)
    plt.ylim(-3,3)

animate(sim_quadratic_potential,init_func=quadratic_init)
    
""" import numpy as np
import matplotlib.pyplot as plt

# Define parameters
L = 5.0   # Width of the potential well
N = 200   # Number of grid points
T = 5.0   # Total simulation time
dt = 0.001 # Time step
h = L/(N-1) # Grid spacing

# Define potential function
def V(x):
    return x**2/2

# Define potential function
def V(x):
    if x < 0 or x > L:
        return np.inf
    else:
        return 0.0 

# Initialize wave function
x = np.linspace(-L/2, L/2, N)
psi = np.exp(-50*(x)**2) # Gaussian wave packet
psi[0] = psi[-1] = 0.0 # Boundary conditions

# Initialize plot
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(x, psi)

# Perform time evolution
for t in np.arange(0, T, dt):
    # Compute second derivative of psi
    d2psi = np.diff(psi, 2)/(h**2)
    d2psi = np.insert(d2psi, 0, 0.0)
    d2psi = np.append(d2psi, 0.0)

    # Compute new wave function
    psi_new = psi + dt*(-1j*d2psi - V(x)*psi)

    # Apply boundary conditions
    psi_new[0] = psi_new[-1] = 0.0

    # Update wave function
    psi = psi_new

    # Update plot
    line.set_ydata(np.abs(psi)**2)
    ax.set_ylim(0, np.max(np.abs(psi)**2))
    fig.canvas.draw()
    fig.canvas.flush_events()

# Keep plot open
plt.ioff()
plt.show() """



""" import numpy as np
import matplotlib.pyplot as plt

# Define simulation parameters
x_min = -10  # Minimum position in simulation
x_max = 10   # Maximum position in simulation
dx = 0.1     # Spatial step size
dt = 0.01    # Time step size
t_max = 5    # Maximum simulation time

# Define the initial wave function
def initial_wave_function(x):
    return np.exp(-x**2)

# Define the potential function
def potential(x):
    return x**2 / 2

# Define the potential function with a small barrier
def potential_bar(x):
    return np.piecewise(x, [x < -1, -1 <= x < 1, x >= 1],
                        [lambda x: 0, lambda x: 0.1, lambda x: 0])

# Initialize the simulation grid
x_grid = np.arange(x_min, x_max, dx)
psi = initial_wave_function(x_grid).astype('complex128')

# Initialize the plot
fig, ax = plt.subplots()
line, = ax.plot(x_grid, np.abs(psi)**2)

# Loop over time steps
for t in np.arange(0, t_max, dt):

    # Calculate the second derivative of psi
    #psi_xx = np.diff(psi, 2) / dx**2
    psi_xx = np.diff(psi, 2)
    psi_xx = np.hstack([psi_xx[0], psi_xx, psi_xx[-1]]) / dx**2

    # Calculate the potential energy
    V = potential_bar(x_grid)

    # Calculate the time derivative of psi
    psi_t = -1j * (V * psi - psi_xx)

    # Update psi using the time derivative
    psi += psi_t * dt

    # Update the plot
    line.set_ydata(np.abs(psi)**2)
    ax.set_title(f"Time = {t:.2f}")
    plt.pause(0.01)

# Show the final plot
plt.show()

%%
import numpy as np
import matplotlib.pyplot as plt

# Define constants and parameters
hbar = 1.0545718e-34   # Planck's constant over 2*pi
m = 9.10938356e-31     # electron mass
L = 1e-8               # length of potential well
N = 1000               # number of spatial grid points
dx = L / N             # spatial step size
x = np.linspace(0, L, N)   # spatial grid
V0 = 10.0 * 1.602e-19  # potential barrier height
a = L / 5              # width of potential barrier
sigma = L / 10         # width of wave packet
kappa = 5e10           # wave vector

# Define the potential energy function
def V(x):
    if x > L/2 - a/2 and x < L/2 + a/2:
        return V0
    else:
        return 0

# Define the initial wave function
def psi0(x):
    return np.exp(-(x-L/2)**2 / (2*sigma**2)) * np.exp(1j * kappa * x)

# Initialize the wave function
psi = psi0(x)

# Define the Hamiltonian operator
def H(psi):
    V_vec = np.vectorize(V)
    Vx = V_vec(x)
    dx2 = dx*dx
    h = hbar*dx2/(2*m)
    c = -hbar**2/(2*m*dx2)
    A = np.eye(N) + 0.5j*h*c*np.diag(Vx)
    B = np.eye(N) - 0.5j*h*c*np.diag(Vx)
    C = np.eye(N) - 0.5j*h*c*np.diag(Vx)
    D = np.eye(N) + 0.5j*h*c*np.diag(Vx)
    E = np.eye(N)
    F = np.eye(N)
    A[0,0] = 1; A[0,1] = 0
    B[0,0] = 1; B[0,1] = 0
    C[0,0] = 1; C[0,1] = 0
    D[0,0] = 1; D[0,1] = 0
    A[-1,-1] = 1; A[-1,-2] = 0
    B[-1,-1] = 1; B[-1,-2] = 0
    C[-1,-1] = 1; C[-1,-2] = 0
    D[-1,-1] = 1; D[-1,-2] = 0
    E[1:,1:] = A[1:,1:] @ np.linalg.inv(B[1:,1:])
    F[1:,1:] = C[1:,1:] @ np.linalg.inv(D[1:,1:])
    Hx = -c*(E - F)
    return Hx @ psi

# Set up the time evolution
t0 = 0                 # initial time
tf = 1e-15             # final time
dt = 1e-19             # time step size
times = np.arange(t
                  
%% """






