import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Use the built-in Streamlit command to render plots inline
st.set_option('deprecation.showPyplotGlobalUse', False)


def run_simulation(m, k, c, F0, Omega, x0, v0, tf, dt):
    """
    Run the Forced Damped Oscillator simulation using 4th-order Runge-Kutta.
    Returns time_points, x_points, and v_points as numpy arrays.
    """

    # Derived parameters
    omega0 = np.sqrt(k / m)          # Natural frequency of the undamped system
    zeta = c / (2.0 * np.sqrt(k * m))  # Damping ratio

    # Define the system of first-order ODEs
    def derivatives(t, x, v):
        dxdt = v
        dvdt = -2*zeta*omega0*v - (omega0**2)*x + (F0/m)*np.cos(Omega*t)
        return dxdt, dvdt

    # One RK4 step
    def rk4_step(t, x, v, dt):
        dxdt1, dvdt1 = derivatives(t, x, v)
        dxdt2, dvdt2 = derivatives(t + 0.5*dt, x + 0.5*dxdt1*dt, v + 0.5*dvdt1*dt)
        dxdt3, dvdt3 = derivatives(t + 0.5*dt, x + 0.5*dxdt2*dt, v + 0.5*dvdt2*dt)
        dxdt4, dvdt4 = derivatives(t + dt,     x + dxdt3*dt,    v + dvdt3*dt)

        x_new = x + (dt / 6.0) * (dxdt1 + 2*dxdt2 + 2*dxdt3 + dxdt4)
        v_new = v + (dt / 6.0) * (dvdt1 + 2*dvdt2 + 2*dvdt3 + dvdt4)
        return x_new, v_new

    # Initialize
    t0 = 0.0
    t = t0
    x = x0
    v = v0

    time_points = [t]
    x_points = [x]
    v_points = [v]

    # Time-stepping with RK4
    while t < tf:
        x, v = rk4_step(t, x, v, dt)
        t += dt
        time_points.append(t)
        x_points.append(x)
        v_points.append(v)

    # Convert to numpy arrays
    time_points = np.array(time_points)
    x_points = np.array(x_points)
    v_points = np.array(v_points)

    return time_points, x_points, v_points


def main():
    st.title("Forced Damped Oscillator Simulation")

    st.markdown(
        """
        This app simulates a **forced damped oscillator** using the 4th-order 
        Runge-Kutta (RK4) method. Adjust the parameters below, then click 
        **Run Simulation**.
        """
    )

    # Sidebar inputs
    st.sidebar.header("Input Parameters")

    # Mass (m)
    m = st.sidebar.number_input("Mass (m) [kg]", value=1.0, format="%.3f")

    # Spring constant (k)
    k = st.sidebar.number_input("Spring constant (k) [N/m]", value=1.0, format="%.3f")

    # Damping coefficient (c)
    c = st.sidebar.number_input("Damping coefficient (c) [N·s/m]", value=0.2, format="%.3f")

    # Forcing amplitude (F0)
    F0 = st.sidebar.number_input("Forcing amplitude (F0) [N]", value=0.5, format="%.3f")

    # Driving frequency (Omega)
    Omega = st.sidebar.number_input("Driving frequency (Ω) [rad/s]", value=1.0, format="%.3f")

    # Initial displacement (x0)
    x0 = st.sidebar.number_input("Initial displacement (x0) [m]", value=0.0, format="%.3f")

    # Initial velocity (v0)
    v0 = st.sidebar.number_input("Initial velocity (v0) [m/s]", value=0.0, format="%.3f")

    # Final time and time step
    tf = st.sidebar.number_input("End time (tf) [s]", value=50.0, format="%.1f")
    dt = st.sidebar.number_input("Time step (dt) [s]", value=0.01, format="%.3f")

    if st.button("Run Simulation"):
        # Run the simulation
        time_points, x_points, v_points = run_simulation(m, k, c, F0, Omega, x0, v0, tf, dt)

        # Plot results
        st.subheader("Displacement vs. Time")
        plt.figure(figsize=(8, 5))
        plt.plot(time_points, x_points, label='Displacement (x)')
        plt.xlabel('Time (s)')
        plt.ylabel('Displacement (m)')
        plt.grid(True)
        plt.legend()
        st.pyplot()

        # Phase plot
        st.subheader("Phase Diagram (Velocity vs. Displacement)")
        plt.figure(figsize=(6, 6))
        plt.plot(x_points, v_points)
        plt.xlabel('Displacement (m)')
        plt.ylabel('Velocity (m/s)')
        plt.grid(True)
        st.pyplot()


if __name__ == "__main__":
    main()
