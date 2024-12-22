# Simulation of a Forced Damped Oscillator using 4th Order Runge-Kutta

YIN-HSUEH YU

## Abstract

In this paper, I present a numerical study of a forced damped harmonic oscillator using a fourth-order Runge-Kutta (RK4) method. The system, governed by a second-order ordinary differential equation, is transformed into a system of first-order ODEs and integrated over time. I analyze the displacement and velocity behavior under different forcing frequencies and damping ratios, and visualize the resulting trajectories and phase portraits. The resulting plots are included. ***I also made an online web application that demonstrates the simulation interactively.***

## 1. Introduction

The forced damped harmonic oscillator is a fundamental model in classical mechanics and engineering, describing a mass-spring-damper system subject to both a resistive damping force and a periodic external driving force:

$$
m \frac{d^2x}{dt^2} + c \frac{dx}{dt} + kx = F_0 \cos(\Omega t),
$$

where $m$ is the mass, $c$ is the damping coefficient, $k$ is the spring constant, $F_0$ is the amplitude of the driving force, and $\Omega$ is the driving frequency.

I introduce dimensionless parameters and rewrite the equation as:

$$
\frac{d^2x}{dt^2} + 2\zeta \omega_0 \frac{dx}{dt} + \omega_0^2 x = \frac{F_0}{m}\cos(\Omega t),
$$

where $\omega_0 = \sqrt{k/m}$ and $\zeta = \frac{c}{2\sqrt{km}}$ is the damping ratio.

## 2. Numerical Method

To solve the equation numerically, we express it as a system of first-order ODEs:

$$
\frac{dx}{dt} = v, \quad \frac{dv}{dt} = -2\zeta\omega_0 v - \omega_0^2 x + \frac{F_0}{m}\cos(\Omega t).
$$

The 4th order Runge-Kutta method is used for numerical integration. Given:

$$
y' = f(t,y),
$$

the RK4 method for a time step $dt$ is:

$$
k_1 = f(t, y), \quad k_2 = f(t+\tfrac{dt}{2}, y + \tfrac{dt}{2}k_1),
$$

$$
k_3 = f(t+\tfrac{dt}{2}, y + \tfrac{dt}{2}k_2), \quad k_4 = f(t+dt, y+dt \cdot k_3),
$$

$$
y_{n+1} = y_n + \frac{dt}{6}(k_1 + 2k_2 + 2k_3 + k_4).
$$

## 3. Implementation in Python

Below is the Python code snippet implementing the forced damped oscillator using the RK4 method:

```python
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
```


## 4. Results

Below are example images of the FDO simulation.
![original image](https://cdn.mathpix.com/snip/images/dFsUnpz2G9bQ7Z5yR4iF2Z04IWrdzq4MXZ4oeeK5AMM.original.fullsize.png)
![original image](https://cdn.mathpix.com/snip/images/Y6Fn5_6xAVFGiJFvnV2kCqfingGTt86Y7PdjscupXUc.original.fullsize.png)

## 5. Interactive Web Application

 I have also created a web-based application that runs this simulation in real time. Users can adjust all the parameters including:mass (kg)
spring constant (N/m)
damping coefficient (N·s/m)
amplitude of the forcing (N)
driving frequency (rad/s)
initial displacement (m)
initial velocity (m/s)
start time (s)
end time (s)
time step (s)
to see how the system behaves under different conditions. Access the interactive app at:
https://rk4.onrender.com/

