import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from scipy.signal import chirp

# --------------------
# Helper Functions
# --------------------
def is_stable(A):
    """Check if A is stable by ensuring all eigenvalues have negative real parts."""
    eig_vals = np.linalg.eigvals(A)
    return np.all(np.real(eig_vals) < 0)

def generate_stable_system(n, eigen_lower=0.1, eigen_upper=2.0):
    """
    Generate a stable state-space system of order n.
    Constructs A using a random orthonormal matrix Q and a diagonal matrix D with negative eigenvalues.
    Also returns random B (nx1) and C (1xn).
    """
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    eigenvalues = -np.random.uniform(eigen_lower, eigen_upper, size=n)
    D = np.diag(eigenvalues)
    A = Q @ D @ Q.T
    if not is_stable(A):
        raise ValueError("Generated system matrix A is not stable.")
    B = np.random.randn(n, 1)
    C = np.random.randn(1, n)
    return A, B, C

def generate_anomalous_system(A, perturbation_factor=0.01, max_attempts=10):
    """
    Generate an anomalous version of A by applying a small random perturbation.
    Repeats until a stable perturbed A is found or raises an error.
    """
    for attempt in range(max_attempts):
        perturbation = perturbation_factor * np.random.randn(*A.shape)
        A_anom = A + perturbation
        if is_stable(A_anom):
            return A_anom
    raise ValueError("Failed to generate a stable anomalous system after multiple attempts.")

def complex_input_generator(offset, f0=10, f1=20, T_total=5.0, noise_std=0.1):
    """
    Create a complex input function that produces a chirp signal.
    The chirp sweeps linearly from f0 to f1 Hz over T_total seconds.
    The chirp amplitude is scaled to 0.9, and additive noise and a DC offset are included.
    """
    def u(t):
        # Scale the chirp amplitude to 0.9
        chirp_signal = 0.9 * chirp(t, f0=f0, t1=T_total, f1=f1, method='linear')
        noise = np.random.normal(0, noise_std)
        return offset + chirp_signal + noise
    return u

def system_dynamics(t, x, A, B, u_func):
    """
    Define the continuous-time dynamics:
        dx/dt = A x + B u(t)
    """
    u_val = u_func(t)
    return A @ x + B.flatten() * u_val

def simulate_system(A, B, C, u_func, T, dt=0.01):
    """
    Simulate the system over time T with timestep dt.
    Returns time array, state trajectories, and output computed as y = C x.
    """
    t_eval = np.arange(0, T, dt)
    x0 = np.zeros(A.shape[0])
    sol = solve_ivp(lambda t, x: system_dynamics(t, x, A, B, u_func),
                    t_span=(0, T), y0=x0, t_eval=t_eval, method='RK45')
    y = (C @ sol.y).flatten()
    return sol.t, sol.y, y

# --------------------
# Simulation Parameters
# --------------------
dt = 0.01
anomaly_runtime = 1001.0  # seconds for anomaly (and test) simulation
normal_runtime = 5001.0   # seconds for normal (training) simulation
drop_anomaly_test = int(1/dt)      # drop first 1 second (for anomaly and test) -> 100 samples
drop_train = int(anomaly_runtime/dt)  # drop first 1001 seconds from training simulation

n_subsystems = 10      # number of subsystems to couple
n_per_subsystem = 6   # each subsystem dimension

# --------------------
# Main Loop for Data Generation
# --------------------
for k in range(1, 101):
    model_name = f"ODE_T{k}"
    # For reproducibility, set a seed per model.
    np.random.seed(k)
    
    # Generate four independent subsystems
    A_blocks, B_list, C_list = [], [], []
    for i in range(n_subsystems):
        A, B, C = generate_stable_system(n_per_subsystem)
        A_blocks.append(A)
        B_list.append(B)
        C_list.append(C)
        
    # Build global system: block-diagonal A
    A_global = block_diag(*A_blocks)
    # Create a coupling matrix with off-diagonal interactions
    coupling_strength = 0.05
    coupling_matrix = coupling_strength * np.random.randn(n_subsystems * n_per_subsystem,
                                                           n_subsystems * n_per_subsystem)
    # Zero out intra-subsystem blocks (ensure coupling only between subsystems)
    for i in range(n_subsystems):
        start = i * n_per_subsystem
        end = start + n_per_subsystem
        coupling_matrix[start:end, start:end] = 0
    A_global += coupling_matrix
    if not is_stable(A_global):
        print(f"Warning: Model {model_name} global system is not stable. Consider reducing coupling strength.")
    
    # Global input and output: single input affects subsystem 1; output from subsystem 1.
    B_global = np.zeros((n_subsystems * n_per_subsystem, 1))
    B_global[:n_per_subsystem, :] = B_list[0]
    C_global = np.zeros((1, n_subsystems * n_per_subsystem))
    C_global[:, :n_per_subsystem] = C_list[0]
    
    # Create the anomalous system (perturb A_global)
    A_global_anom = generate_anomalous_system(A_global, perturbation_factor=0.01)
    
    # Define directories for saving data
    base_dir = os.path.join(".", "data", f"model_T{k}")
    dir_anom = os.path.join(base_dir, "anom")
    dir_test = os.path.join(base_dir, "test")
    dir_train = os.path.join(base_dir, "train")
    for d in [dir_anom, dir_test, dir_train]:
        os.makedirs(d, exist_ok=True)
    
    # Loop over simulation types and sine offsets
    for currentVar in ["anomalies", "test", "train"]:
        for sine_offset in np.arange(0, 22, 2):  # 0,2,...,20
            # For each run, create an input function with the given sine_offset.
            # Note: T_total in the input generator is used for the chirp sweep.
            if currentVar in ["anomalies", "test"]:
                T_total_input = anomaly_runtime
            else:
                T_total_input = normal_runtime
            u_func = complex_input_generator(offset=sine_offset, f0=10, f1=20,
                                             T_total=T_total_input, noise_std=0.1)
            
            if currentVar == "anomalies":
                # Simulate the anomalous system for anomaly_runtime seconds.
                t_sim, _, y_sim = simulate_system(A_global_anom, B_global, C_global, u_func,
                                                  T=anomaly_runtime, dt=dt)
                # Compute input values
                u_values = np.array([u_func(t) for t in t_sim])
                # Remove the first 1 second of samples.
                t_sim = t_sim[drop_anomaly_test:]
                u_values = u_values[drop_anomaly_test:]
                y_sim = y_sim[drop_anomaly_test:]
                csv_name = os.path.join(dir_anom, f"{model_name}_{sine_offset}_anom.csv")
            
            elif currentVar == "test":
                # Simulate normal system for anomaly_runtime seconds to generate test data.
                t_sim, _, y_sim = simulate_system(A_global, B_global, C_global, u_func,
                                                  T=anomaly_runtime, dt=dt)
                u_values = np.array([u_func(t) for t in t_sim])
                t_sim = t_sim[drop_anomaly_test:]
                u_values = u_values[drop_anomaly_test:]
                y_sim = y_sim[drop_anomaly_test:]
                csv_name = os.path.join(dir_test, f"{model_name}_{sine_offset}_OK.csv")
            
            elif currentVar == "train":
                # Simulate normal system for normal_runtime seconds (training data).
                t_sim, _, y_sim = simulate_system(A_global, B_global, C_global, u_func,
                                                  T=normal_runtime, dt=dt)
                u_values = np.array([u_func(t) for t in t_sim])
                t_sim = t_sim[drop_train:]
                u_values = u_values[drop_train:]
                y_sim = y_sim[drop_train:]
                csv_name = os.path.join(dir_train, f"{model_name}_{sine_offset}_OK.csv")
            
            # Create a DataFrame with time, input, and output.
            df = pd.DataFrame({"Input": u_values, "Output": y_sim})
            df.to_csv(csv_name, index=False)
            print(f"Saved {csv_name}")
