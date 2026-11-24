import numpy as np

def analisar_dados_viciados(tabela_conjunta):
  soma_total = np.sum(tabela_conjunta)
  assert np.isclose(soma_total, 1.0), f"Erro: soma deu {soma_total}, deveria ser 1.0"

  marginal_x = np.sum(tabela_conjunta, axis=1)
  marginal_y = np.sum(tabela_conjunta, axis=0)

  faces = [1, 2, 3, 4, 5, 6]
  cabecalho = "X \\ Y | " + " | ".join(f"  Y={y}  " for y in faces) + " | Marginal P(X)"
  print(cabecalho)
  print("-" * len(cabecalho))

  for i, x in enumerate(faces):
    linha_vals = " | ".join(f"{tabela_conjunta[i, j]:7.4f}" for j in range(6))
    print(f" X={x}  | {linha_vals} |    {marginal_x[i]:.4f}")

  rodape_vals = " | ".join(f"{marginal_y[j]:7.4f}" for j in range(6))
  print(f"P(Y)  | {rodape_vals} |    {soma_total:.4f}\n")

  # Calcula P(X = x) * P(Y = y) para todos os pares (x, y) via produto externo
  produto_marginais = np.outer(marginal_x, marginal_y)
  sao_independentes = np.allclose(tabela_conjunta, produto_marginais)
  print(f"Os dados são independentes? -> {'SIM' if sao_independentes else 'NÃO'}")

  if not sao_independentes:
    diferencas = np.abs(tabela_conjunta - produto_marginais)
    x_idx, y_idx = np.unravel_index(np.argmax(diferencas), diferencas.shape)
    print(
        f"Exemplo de falha em (X={x_idx+1}, Y={y_idx+1}): "
        f"P(X,Y) = {tabela_conjunta[x_idx, y_idx]:.4f} != "
        f"P(X)*P(Y) = {produto_marginais[x_idx, y_idx]:.4f}"
    )
  print("\n")

p_x = np.array([0.05, 0.10, 0.10, 0.15, 0.25, 0.35])
p_y = np.array([0.30, 0.25, 0.20, 0.10, 0.10, 0.05])

# Como são independentes, a conjunta é exatamente o produto externo P(X) * P(Y)
conjunta_independente = np.outer(p_x, p_y)
analisar_dados_viciados(conjunta_independente)