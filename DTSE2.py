import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define constants
L = 1  # Length of the system
dx = 0.01  # Spatial step size
dt = 0.01  # Time step size
N = int(L/dx) + 1  # Number of spatial points
t_max = 2  # Maximum time
k0 = 100  # Wave number of the wave packet
m = 1  # Mass of the particle
hbar = 1  # Planck's constant

# Define potential barrier
V0 = 1000  # Potential barrier height
a = 0.5  # Barrier width
V = np.zeros(N)
for i in range(N):
    if L/2 +a/2-0.05 <= i*dx <= L/2 + a/2:
        V[i] = V0

# Define initial wave function
x = np.linspace(0, L, N)
psi0 = np.exp(-(x-L/2)**2/(2*(0.05*L)**2))*np.exp(1j*k0*x)

# Define matrices for solving Schrodinger's equation
I = np.identity(N)
D2 = (np.diag(np.ones(N-1), 1) - 2*np.identity(N) + np.diag(np.ones(N-1), -1))/(dx**2)
H = -(hbar**2)/(2*m)*D2 + np.diag(V)

# Define function to update the wave function at each time step
def update_psi(psi, H):
    psi_next = np.linalg.solve(I + 0.5j*dt/hbar*H, (I - 0.5j*dt/hbar*H)@psi)
    return psi_next

# Set up the plot
fig, ax = plt.subplots()
line, = ax.plot(x, np.abs(psi0)**2)
ax.set_xlabel('Position')
ax.set_ylabel('Probability Density')
ax.set_title('Time-Dependent Wave Function')

# Define function to update the plot at each time step
def update_plot(frame, psi):
    psi_next = update_psi(psi, H)
    line.set_ydata(np.abs(psi_next)**2)
    psi[:] = psi_next[:]
    return line,

# Create the animation
psi = psi0
ani = animation.FuncAnimation(fig, update_plot, frames=np.arange(0, t_max, dt), fargs=(psi,), blit=True)

# Add the potential barrier to the plot
barrier = ax.fill_between(x, 0, V, alpha=0.2, color='gray')

# Show the plot
plt.show()