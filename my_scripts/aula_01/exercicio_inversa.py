import math
import random
import matplotlib.pyplot as plt

def exponential_pdf(x, lam):
    if x < 0:
        return 0
    return lam * math.exp(-lam * x)

def sample_exponential(lam, n_amostras):
    amostras = []
    for i in range(1, n_amostras):
        u = random.random()
        x = -math.log(1-u) / lam
        amostras.append(x)
    return amostras

lam = 2.0
n_amostras = 5000
random.seed(42)

amostras = sample_exponential(lam, n_amostras)
x_max = max(amostras)
n_pontos = 200
passo = x_max / n_pontos
x_vals = [i * passo for i in range(n_pontos+1)]
y_vals = [exponential_pdf(x, lam) for x in x_vals]

plt.figure(figsize=(9, 5))
plt.hist(
    amostras,
    bins=50,
    density=True,
    alpha=0.6,
    color="steelblue",
    edgecolor="black",
    label="Amostras (Histograma)",
)

plt.plot(
    x_vals,
    y_vals,
    color="crimson",
    linewidth=2.5,
    label=f"PDF Verdadeira ($\\lambda = {lam}$)",
)
plt.savefig("exponencial_transformada_inversa.png", dpi=150)
plt.show()
