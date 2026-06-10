import os
import numpy as np

def main():
    # Percorso della cartella data
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    params_path = os.path.join(data_dir, "params_250.npy")
    codes_path = os.path.join(data_dir, "pod_codes_250.npy")
    traj_path = os.path.join(data_dir, "trajectories_250.npy")
    
    # Caricamento dei dati
    params = np.load(params_path)
    codes = np.load(codes_path)
    trajectories = np.load(traj_path)
    
    print("=== ANTEPRIMA PARAMETRI (Prime 5 righe) ===")
    print("Colonne: [Velocità iniziale, Attrito, Distanza ostacolo, Ritardo nominale]")
    print(np.round(params[:5], 3))
    
    print("\n=== ANTEPRIMA CODICI POD (Prime 5 righe) ===")
    print(f"Formato: {codes.shape} (N_scenari, N_modi)")
    print(np.round(codes[:5], 3))
    
    print("\n=== INFO TRAIETTORIE ===")
    print(f"Formato: {trajectories.shape} -> (250 scenari, {trajectories.shape[1]} step temporali, 2 valori: [posizione, velocità])")
    print("Essendo i dati delle traiettorie in 3D (3 dimensioni), non sono esportabili in un singolo CSV classico.")
    print("Esempio: Stato finale del PRIMO scenario [posizione (m), velocità (m/s)]:")
    print(np.round(trajectories[0, -1, :], 3))
    
    # Salvataggio in formato CSV
    params_csv = os.path.join(data_dir, "params_250.csv")
    codes_csv = os.path.join(data_dir, "pod_codes_250.csv")
    
    np.savetxt(params_csv, params, delimiter=",", header="initial_speed,friction,detection_dist,nominal_delay", comments="")
    np.savetxt(codes_csv, codes, delimiter=",", header=",".join([f"mode_{i}" for i in range(codes.shape[1])]), comments="")
    
    print(f"\n✅ Ho esportato i file in formato CSV (apribili con Excel) in:\n- {params_csv}\n- {codes_csv}")

if __name__ == "__main__":
    main()
