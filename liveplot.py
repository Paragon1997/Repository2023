import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Create a figure and axis object
fig, ax = plt.subplots()

# Define the x and y data arrays
x_data = np.linspace(0, 2*np.pi, 100)
y_data = np.sin(x_data)

# Create a line object to plot the data
line, = ax.plot(x_data, y_data)

# Define the update function that is called by FuncAnimation
def update(frame):
    # Calculate the new y data based on the current frame
    y_data = np.sin(x_data + frame/10)
    
    # Update the line data
    line.set_ydata(y_data)
    
    # Return the line object
    return line,

# Create the animation object
animation = FuncAnimation(fig, update, frames=np.arange(0, 100), interval=50)

# Show the plot
plt.show()