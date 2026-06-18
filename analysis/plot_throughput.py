import matplotlib.pyplot as plt

# Dados reais coletados nos seus benchmarks
req_per_sec = [1000, 2300, 2500, 2600, 3000]
throughput = [19418.74, 30133.33, 31984.31, 32421.35, 32288.94]

plt.style.use('dark_background')
plt.figure(figsize=(12, 7))

# Plotando a curva de throughput
plt.plot(req_per_sec, throughput, marker='D', markersize=8, color='#ff00ff', label='Measured Throughput', linewidth=3)

# Linha de Tendência Ideal (O que seria 100% linear)
plt.plot([1000, 3000], [19418, 32421], linestyle='--', color='gray', alpha=0.5, label='Ideal Scaling')

# Destaque para a Saturação do Silício
plt.axhline(y=32400, color='yellow', linestyle=':', alpha=0.7)
plt.text(1050, 33000, 'Hardware Compute Limit (RTX 2070)', color='yellow', fontweight='bold')

# Estética
plt.title('Mini-LLM Throughput Analysis: Load vs. Output\n(Stories110M Model)', fontsize=16, color='white', pad=20)
plt.xlabel('Input Load (Poisson Requests per Second)', fontsize=12)
plt.ylabel('Token Throughput (tok/s)', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.2)
plt.legend(fontsize=12)

# Salva
plt.tight_layout()
plt.savefig('throughput_scaling_curve.png', dpi=300)
print("✅ Gráfico 'throughput_scaling_curve.png' gerado!")

