### Seismocardiogram smartphone data processing
### loads data from excel with gravitationa acceleration
### calculates the grav-vector
### rotates data to align z with grav-vector
### removes grav-vector from measurements
### Note: code from Chat GPT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS
# ============================================================

FILE = "seismokardio signal w g 2 010626 1441 2026-06-01 14-42-25.xls"

TIME_COLUMN = "Time (s)"

ACCEL_X_COLUMN = "Acceleration x (m/s^2)"
ACCEL_Y_COLUMN = "Acceleration y (m/s^2)"
ACCEL_Z_COLUMN = "Acceleration z (m/s^2)"

CUT_FIRST_SECONDS = 3.0

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_excel(FILE)

time = df[TIME_COLUMN].values

ax = df[ACCEL_X_COLUMN].values
ay = df[ACCEL_Y_COLUMN].values
az = df[ACCEL_Z_COLUMN].values

# ============================================================
# REMOVE FIRST 3 SECONDS
# ============================================================

mask = time >= CUT_FIRST_SECONDS

time = time[mask]

ax = ax[mask]
ay = ay[mask]
az = az[mask]

time = time - time[0]

# ============================================================
# PLOT RAW ACCELEROMETER SIGNALS
# ============================================================

plt.figure(figsize=(12,5))

plt.plot(time, ax, label='ax')
plt.plot(time, ay, label='ay')
plt.plot(time, az, label='az')

plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s²]")
plt.title("Raw Accelerometer Data Including Gravity")

plt.legend()
plt.grid()

plt.show()

# ============================================================
# ESTIMATE GRAVITY VECTOR
# ============================================================

g_vector = np.array([
    np.mean(ax),
    np.mean(ay),
    np.mean(az)
])

g_magnitude = np.linalg.norm(g_vector)

vertical_unit_vector = g_vector / g_magnitude

print()
print("Estimated gravity vector:")
print(g_vector)

print()
print("Estimated |g|:")
print(g_magnitude)

print()
print("Vertical unit vector:")
print(vertical_unit_vector)

# ============================================================
# PROJECT ACCELERATION ONTO TRUE VERTICAL
# ============================================================

accel_matrix = np.column_stack((ax, ay, az))

vertical_acceleration_raw = accel_matrix @ vertical_unit_vector

# ============================================================
# PLOT TRUE VERTICAL ACCELERATION
# ============================================================

plt.figure(figsize=(12,5))

plt.plot(time, vertical_acceleration_raw)

plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s²]")

plt.title("Acceleration Along Estimated Vertical Direction")

plt.grid()

plt.show()

# ============================================================
# REMOVE GRAVITY
# ============================================================

vertical_linear_acceleration = (
    vertical_acceleration_raw
    - np.mean(vertical_acceleration_raw)
)

# ============================================================
# PLOT VERTICAL LINEAR ACCELERATION
# ============================================================

plt.figure(figsize=(12,5))

plt.plot(time, vertical_linear_acceleration)

plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s²]")

plt.title("Vertical Linear Acceleration (Gravity Removed)")

plt.grid()

plt.show()