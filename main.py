import numpy as np
import matplotlib.pyplot as plt

# Parameters for the forced damped oscillator
m = 1.0         # mass (kg)
k = 1.0         # spring constant (N/m)
c = 0.2         # damping coefficient (N·s/m)
F0 = 0.5        # amplitude of the forcing (N)
Omega = 1.0      # driving frequency (rad/s)

# Initial conditions
x0 = 0.0  # initial displacement (m)
v0 = 0.0  # initial velocity (m/s)

# Time parameters
t0 = 0.0    # start time (s)
tf = 50.0   # end time (s)
dt = 0.01   # time step (s)

# Derived parameters
# natural frequency of undamped system
omega0 = np.sqrt(k/m)
# damping ratio zeta = c/(2*sqrt(k*m))
zeta = c/(2.0*np.sqrt(k*m))

# Define the system of first-order ODEs
def derivatives(t, x, v):
    # dx/dt = v
    dxdt = v
    # dv/dt = - (2*zeta*omega0)*v - (omega0^2)*x + (F0/m)*cos(Omega*t)
    dvdt = -2*zeta*omega0*v - (omega0**2)*x + (F0/m)*np.cos(Omega*t)
    return dxdt, dvdt

# Implement one RK4 step
def rk4_step(t, x, v, dt):
    # k1
    dxdt1, dvdt1 = derivatives(t, x, v)
    # k2
    dxdt2, dvdt2 = derivatives(t + 0.5*dt, x + 0.5*dxdt1*dt, v + 0.5*dvdt1*dt)
    # k3
    dxdt3, dvdt3 = derivatives(t + 0.5*dt, x + 0.5*dxdt2*dt, v + 0.5*dvdt2*dt)
    # k4
    dxdt4, dvdt4 = derivatives(t + dt, x + dxdt3*dt, v + dvdt3*dt)

    x_new = x + (dt/6.0)*(dxdt1 + 2*dxdt2 + 2*dxdt3 + dxdt4)
    v_new = v + (dt/6.0)*(dvdt1 + 2*dvdt2 + 2*dvdt3 + dvdt4)

    return x_new, v_new

# Initialize arrays for storage
time_points = [t0]
x_points = [x0]
v_points = [v0]

# Time-stepping with RK4
t = t0
x = x0
v = v0

while t < tf:
    x, v = rk4_step(t, x, v, dt)
    t += dt
    time_points.append(t)
    x_points.append(x)
    v_points.append(v)

# Convert results to numpy arrays for convenience
time_points = np.array(time_points)
x_points = np.array(x_points)
v_points = np.array(v_points)

# Plot the displacement over time
plt.figure(figsize=(10,6))
plt.plot(time_points, x_points, label='Displacement (x)')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.title('Forced Damped Oscillation (RK4)')
plt.grid(True)
plt.legend()
plt.show()

# Optional: Phase plot (v vs x)
plt.figure(figsize=(6,6))
plt.plot(x_points, v_points)
plt.xlabel('Displacement (m)')
plt.ylabel('Velocity (m/s)')
plt.title('Phase Diagram')
plt.grid(True)
plt.show()
