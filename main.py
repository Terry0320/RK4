import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

# We'll keep the derivatives and RK4 step inside a function for cleanliness.
def simulate_fdo(m, k, c, F0, Omega, x0, v0, t0, tf, dt):
    """
    Solve the forced damped oscillator using RK4 and return two Matplotlib plots:
    1) Displacement vs. Time
    2) Phase Diagram (Velocity vs. Displacement)
    """

    # --- Define helper functions ---
    # natural frequency of undamped system
    omega0 = np.sqrt(k/m)
    # damping ratio
    zeta = c / (2.0 * np.sqrt(k * m))

    def derivatives(t, x, v):
        """
        Returns dx/dt, dv/dt for the forced damped oscillator.
        """
        dxdt = v
        dvdt = -2*zeta*omega0*v - (omega0**2)*x + (F0/m)*np.cos(Omega*t)
        return dxdt, dvdt

    def rk4_step(t, x, v, dt):
        dxdt1, dvdt1 = derivatives(t, x, v)

        dxdt2, dvdt2 = derivatives(t + 0.5*dt, x + 0.5*dxdt1*dt, v + 0.5*dvdt1*dt)
        dxdt3, dvdt3 = derivatives(t + 0.5*dt, x + 0.5*dxdt2*dt, v + 0.5*dvdt2*dt)
        dxdt4, dvdt4 = derivatives(t + dt,     x + dxdt3*dt,     v + dvdt3*dt)

        x_new = x + (dt/6.0)*(dxdt1 + 2*dxdt2 + 2*dxdt3 + dxdt4)
        v_new = v + (dt/6.0)*(dvdt1 + 2*dvdt2 + 2*dvdt3 + dvdt4)
        return x_new, v_new

    # --- Perform the numerical integration with RK4 ---
    time_points = [t0]
    x_points = [x0]
    v_points = [v0]

    t = t0
    x = x0
    v = v0

    while t < tf:
        x, v = rk4_step(t, x, v, dt)
        t += dt
        time_points.append(t)
        x_points.append(x)
        v_points.append(v)

    # Convert lists to numpy arrays
    time_points = np.array(time_points)
    x_points = np.array(x_points)
    v_points = np.array(v_points)

    # --- Create the plots ---
    # 1) Displacement vs. time
    fig_time = plt.figure(figsize=(8, 5))
    plt.plot(time_points, x_points, label='Displacement (x)')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.title('Forced Damped Oscillation (RK4)')
    plt.grid(True)
    plt.legend()

    # 2) Phase plot
    fig_phase = plt.figure(figsize=(5, 5))
    plt.plot(x_points, v_points, label='Phase Trajectory')
    plt.xlabel('Displacement (m)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Phase Diagram')
    plt.grid(True)
    plt.legend()

    # Return the figures
    return fig_time, fig_phase

# Build the Gradio interface
demo = gr.Interface(
    fn=simulate_fdo,
    inputs=[
        gr.Number(value=1.0, label="Mass (m)"),
        gr.Number(value=1.0, label="Spring Constant (k)"),
        gr.Number(value=0.2, label="Damping Coefficient (c)"),
        gr.Number(value=0.5, label="Forcing Amplitude (F0)"),
        gr.Number(value=1.0, label="Driving Frequency (Omega)"),
        gr.Number(value=0.0, label="Initial Displacement (x0)"),
        gr.Number(value=0.0, label="Initial Velocity (v0)"),
        gr.Number(value=0.0, label="Start Time (t0)"),
        gr.Number(value=50.0, label="End Time (tf)"),
        gr.Number(value=0.01, label="Time Step (dt)")
    ],
    outputs=[
        gr.Plot(label="Time Series: x(t)"),
        gr.Plot(label="Phase Plot: v vs. x")
    ],
    title="Forced Damped Oscillator (RK4)",
    description=(
        "This app numerically solves the Forced Damped Oscillator problem using the "
        "RK4 method. Adjust parameters and see the displacement over time and the "
        "phase plot."
    )
)

if __name__ == "__main__":
    demo.launch()
