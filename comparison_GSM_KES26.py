import numpy as np

import matplotlib.pyplot as plt
from gsm_r import gsm_original_nd, gsm_rank_based_nd

# =========================================================
# 1. Función objetivo: Rastrigin (vectorizada)
# =========================================================
def rastrigin(X: np.ndarray) -> np.ndarray:
    """
    Rastrigin function, mínimo global en f(0,...,0) = 0.
    Dominio típico: [-5.12, 5.12]^D
    """
    A = 10.0
    return A * X.shape[1] + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1)

# =========================================================
# 2. Configuración del experimento
# =========================================================
D = 100                # Dimensiones
LOWB = np.full(D, -5.12)
UPB  = np.full(D,  5.12)
N = 50                # Tamaño de población
T = 1000              # Iteraciones (generaciones)
SEED = 42

rng = np.random.default_rng(SEED)

# Población inicial compartida para comparación justa
r0 = LOWB + (UPB - LOWB) * rng.random((N, D))

# Z: perturbación gaussiana usada por GSM en cada generación
# (puedes ajustar la escala; 0.1 es un valor conservador)
Z = rng.standard_normal((T, D)) * 0.1

print("=" * 60)
print("GSM vs GSM-R  |  Rastrigin 50D")
print("=" * 60)
print(f"Población : {N}")
print(f"Iteraciones : {T}")
print(f"Semilla : {SEED}")
print()

# =========================================================
# 3. Ejecución
# =========================================================
print("Ejecutando GSM Original...")
best_gsm, hist_gsm = gsm_original_nd(
    r0, Z, LOWB, UPB, rastrigin,
    N=N, T=T,
    gamma=0.10, eps=0.10, Is=5, rho=0.03,
    return_hist=True
)

print("Ejecutando GSM-R (Rank-Based)...")
best_gsmr, hist_gsmr = gsm_rank_based_nd(
    r0, Z, LOWB, UPB, rastrigin,
    N=N, T=T,
    gamma=0.12, eps=0.30, Is=5, rho=0.10, alpha=3.0, invert=True,
    return_hist=True
)

# =========================================================
# 4. Resultados
# =========================================================
print("\n" + "=" * 60)
print("RESULTADOS")
print("=" * 60)
print(f"GSM   best final : {best_gsm:.6e}")
print(f"GSM-R best final : {best_gsmr:.6e}")

# =========================================================
# 5. Gráfica de convergencia
# =========================================================
plt.figure(figsize=(10, 5))
plt.semilogy(hist_gsm,  label="GSM Original",  linewidth=2)
plt.semilogy(hist_gsmr, label="GSM-R Rank-Based", linewidth=2)
plt.xlabel("Iteración", fontsize=12)
plt.ylabel("Mejor fitness (log scale)", fontsize=12)
plt.title("Convergencia en Rastrigin 50D", fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("convergence_gsm_rastrigin.png", dpi=150)
plt.show()

print("\nGráfica guardada: convergence_gsm_rastrigin.png")