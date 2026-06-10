"""
Step 2.1 test - Validate the POD Embedder.

Generates >200 trajectories using the Emergency Braking Simulator,
fits the EmbedderPOD with a 99% variance threshold, and checks
if the reconstruction error meets the criteria.
"""

import sys
import os
import numpy as np

# Risaliamo di 3 livelli: test -> simulators -> root directory del progetto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulators.emergency_braking import EmergencyBrakingSimulator
from embedder.pod import EmbedderPOD
from scipy.stats.qmc import LatinHypercube, scale

def main():
    print("--- Testing POD Embedder ---")
    
    # 1. Generate 250 trajectories (Dataset requirement)
    sim = EmergencyBrakingSimulator()
    bounds = sim.ParamBounds()
    
    N_SAMPLES = 250
    sampler = LatinHypercube(d=4, seed=42)
    unit_samples = sampler.random(n=N_SAMPLES)
    params = scale(unit_samples, bounds["lower"], bounds["upper"])
    
    print(f"Generating {N_SAMPLES} trajectories...")
    trajectories = sim.run(params)  # shape (N, T, 2)
    
    # 2. Fit the POD embedder
    print("\nFitting EmbedderPOD with 99% variance threshold...")
    pod = EmbedderPOD(variance_threshold=0.99)
    pod.fit(trajectories)
    
    print(f"Original trajectories shape : {trajectories.shape}")
    print(f"Number of modes selected  : {pod.nModes} (out of {trajectories.shape[1] * 2})")
    print(f"Actual explained variance : {pod.explained_variance * 100:.4f}%")
    
    # 3. Test embed and reconstruct
    codes = pod.embed(trajectories)
    print(f"Embedded codes shape      : {codes.shape}")
    
    reconstructed = pod.reconstruct(codes)
    
    # 4. Check error
    mse = np.mean((trajectories - reconstructed) ** 2)
    max_err = np.max(np.abs(trajectories - reconstructed))
    print(f"\nReconstruction MSE        : {mse:.6f}")
    print(f"Reconstruction Max Error  : {max_err:.6f}")
    
    # 5. Save the datasets to a folder (inside the 'test' directory)
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    orig_path = os.path.join(data_dir, "trajectories_250.npy")
    codes_path = os.path.join(data_dir, "pod_codes_250.npy")
    params_path = os.path.join(data_dir, "params_250.npy")
    
    np.save(orig_path, trajectories)
    np.save(codes_path, codes)
    np.save(params_path, params)
    
    print(f"\nDatasets salvati con successo nella cartella: {data_dir}")
    print(f"- Originale (250, T, 2): {orig_path}")
    print(f"- Ottimizzato (250, nModes): {codes_path}")
    print(f"- Parametri (250, 4): {params_path}")

if __name__ == "__main__":
    main()