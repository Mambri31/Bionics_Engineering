import csv
import matplotlib.pyplot as plt
import os 
import numpy as np

script_dir = os.path.dirname(__file__) 


file_path_1 = os.path.join(script_dir, "dati_1.csv")
file_path_2 = os.path.join(script_dir, "dati_2.csv")
file_path_3 = os.path.join(script_dir, "dati_3.csv")

# Funzione dati
def carica_dati(path):
    if not os.path.exists(path):
        print(f"Warning: file {path} not found")
        return np.array([]) 
    
    with open(path, "r") as f:
        reader = csv.reader(f)
        dati = list(reader)
    return np.array(dati, dtype=float)

# Caricamento matrici 
matrice1 = carica_dati(file_path_1)
matrice2 = carica_dati(file_path_2)
matrice3 = carica_dati(file_path_3)

if matrice1.size == 0 and matrice2.size == 0 and matrice3.size == 0:
    print("Errore: nessun dato caricato.")
    exit()


# Funzione plot

def plotta_iterazione(matrice, num_figura, titolo_grafico, nome_output):
    if matrice.size == 0:
        return

    # Estrazione Dati
    errX = matrice[:,0]
    errY = matrice[:,1]
    joint0 = matrice[:,2] 
    joint2 = matrice[:,3] 
    joint3 = matrice[:,4] 
    joint4 = matrice[:,5] 

    MAE_x = np.mean(np.abs(errX))
    MAE_y = np.mean(np.abs(errY))
    t = np.linspace(0, 15, errX.size)

   
    plt.figure(num_figura, figsize=(12, 10))

    # Subplot 1 error X
    plt.subplot(2,2,1)
    plt.plot(t, errX)
    plt.xlabel("Time [s]")
    plt.ylabel("Error [pixel]")
    plt.grid(True)
    plt.title(f"{titolo_grafico}: MAE x = {MAE_x:.1f} [px]")

    # Subplot 2 error Y
    plt.subplot(2,2,2)
    plt.plot(t, errY)
    plt.xlabel("Time [s]")
    plt.ylabel("Error [pixel]")
    plt.grid(True)
    plt.title(f"{titolo_grafico}: MAE y = {MAE_y:.1f} [px]")

    # Subplot 3 (Vertical Joints)
    plt.subplot(2,2,4)
    plt.plot(t, joint0, label="Neck pitch")
    plt.plot(t, joint3, label="Eyes tilt")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [°]")
    plt.grid(True)
    plt.legend()
    plt.title("Up and down Movement")

    # Subplot 4 (Horizontal Joints)
    plt.subplot(2,2,3)
    plt.plot(t, joint2, label="Neck yaw")
    plt.plot(t, joint4, label="Eyes pan")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [°]")
    plt.grid(True)
    plt.legend()
    plt.title("Left and right movement")
    
    
    
    plt.tight_layout()
    
   
    percorso_salvataggio = os.path.join(script_dir, nome_output)
    plt.savefig(percorso_salvataggio, dpi=300, bbox_inches='tight')
    print(f"Grafico salvato in: {percorso_salvataggio}")


# Generazione Grafici



plotta_iterazione(matrice1, 1, "FILE 1 (Iterazione 1)", "grafico_iterazione_1.png")
plotta_iterazione(matrice2, 2, "FILE 2 (Iterazione 2)", "grafico_iterazione_2.png")
plotta_iterazione(matrice3, 3, "FILE 3 (Iterazione 3)", "grafico_iterazione_3.png")

plt.show() 
