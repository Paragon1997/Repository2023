import numpy as np
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

# Initialize wave function
x = np.linspace(-L/2, L/2, N)
psi = np.exp(-50*(x)**2) # Gaussian wave packet
psi[0] = psi[-1] = 0.0 # Boundary conditions

# Normalize wave function
psi = psi/np.sqrt(np.trapz(np.abs(psi)**2, x))

# Initialize plot
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(x, np.abs(psi)**2)

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

    # Normalize wave function
    psi_new = psi_new/np.sqrt(np.trapz(np.abs(psi_new)**2, x))

    # Update wave function
    psi = psi_new

    # Update plot
    line.set_ydata(np.abs(psi)**2)
    ax.set_ylim(0, np.max(np.abs(psi)**2))
    fig.canvas.draw()
    fig.canvas.flush_events()

# Keep plot open
plt.ioff()
plt.show()
