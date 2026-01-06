import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Constants
earth_radius = 637.1  # Earth's radius scaled down to 10%

# Orbit parameters
altitude1 = 205
inclination1 = 55.9328
altitude2 = 741
inclination2 = 98.2

# Generate the sphere (Earth)
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = earth_radius * np.outer(np.cos(u), np.sin(v))
y = earth_radius * np.outer(np.sin(u), np.sin(v))
z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

# Function to generate orbit
def generate_orbit(altitude, inclination):
    r = 6371 + altitude  # Use the original Earth radius for orbits
    theta = np.linspace(0, 2 * np.pi, 100)
    orbit_x = r * np.cos(theta)
    orbit_y = r * np.sin(theta)
    orbit_z = np.zeros_like(theta)

    # Rotation matrix for inclination
    inclination_rad = np.radians(inclination)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(inclination_rad), -np.sin(inclination_rad)],
        [0, np.sin(inclination_rad), np.cos(inclination_rad)]
    ])

    # Apply rotation
    orbit_coords = np.vstack((orbit_x, orbit_y, orbit_z))
    rotated_orbit = rotation_matrix @ orbit_coords

    return rotated_orbit[0], rotated_orbit[1], rotated_orbit[2]

# Generate orbits
orbit1_x, orbit1_y, orbit1_z = generate_orbit(altitude1, inclination1)
orbit2_x, orbit2_y, orbit2_z = generate_orbit(altitude2, inclination2)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot Earth
ax.plot_surface(x, y, z, color='b', alpha=0.6)

# Plot orbits
ax.plot(orbit1_x, orbit1_y, orbit1_z, color='r', label=f'Parking Orbit (Altitude: {altitude1} km, Inclination: {inclination1}°)')
ax.plot(orbit2_x, orbit2_y, orbit2_z, color='g', label=f'Sun Synchronous Orbit (Altitude: {altitude2} km, Inclination: {inclination2}°)')

# Labels and legend
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('3D Plot of Earth and Orbits')
ax.legend()

# Set aspect ratio to be equal
ax.set_box_aspect([1, 1, 1])

def update(angle):
    ax.view_init(elev=30, azim=angle)

# Create animation
# ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 1), interval=50)

plt.show()
