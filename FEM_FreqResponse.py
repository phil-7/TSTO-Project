import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

# Parameters
num_segments = 11  # Number of beam segments
r = 1.4142136  # Radius (m)
t = 0.001065  # Wall thickness (m)
E = 72400e6  # Young's modulus (N/m²)
m_added = 3974  # Added mass per unit length (kg/m)
m_total = 42557  # Total mass of LV (kg)
LV_length = 27.86  # Total length of LV (m)

# Calculate moment of inertia
I = (np.pi / 4) * r**4

# Calculate segment length
L = LV_length / num_segments

# Assemble stiffness matrix K
K_segment = (E * I) / L**3 * np.array([[12, 6], [6, 4]])
K_global = np.zeros((2*num_segments, 2*num_segments))
for i in range(num_segments):
    K_global[2*i:2*i+2, 2*i:2*i+2] = K_segment

# Assemble mass matrix M (assuming uniform mass distribution)
M_segment = m_added * L * np.array([[1, 0], [0, 1]])
M_global = np.zeros((2*num_segments, 2*num_segments))
for i in range(num_segments):
    M_global[2*i:2*i+2, 2*i:2*i+2] = M_segment

# Solve eigenvalue problem
eigenvals, eigenvecs = np.linalg.eig(np.dot(np.linalg.inv(M_global), K_global))

# Extract first six non-zero frequencies
num_modes = 6
frequencies = np.sqrt(eigenvals)
first_six_frequencies = frequencies[:num_modes]

print("First six non-zero frequencies:")
print(first_six_frequencies)

# Damping ratio
zeta = 0.05

# Compute modified frequency response
w, h = freqz(b=K_global[0, 0], a=np.array([M_global[0, 0], 2*zeta*frequencies[0], frequencies[0]**2]), worN=200)
magnitude_response = np.abs(h)
phase_response = np.angle(h)

# Plot frequency response
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.semilogx(w, 20 * np.log10(magnitude_response))
plt.title("Frequency Response")
plt.xlabel("Frequency (rad/s)")
plt.ylabel("Magnitude (dB)")

plt.subplot(2, 1, 2)
plt.semilogx(w, phase_response)
plt.xlabel("Frequency (rad/s)")
plt.ylabel("Phase (radians)")

plt.tight_layout()
plt.show()
