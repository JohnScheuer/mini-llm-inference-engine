import matplotlib.pyplot as plt

# Dados reais do seu benchmark
req_per_sec = [1000, 2300, 2500, 2600]
p99_latency = [134.4, 134.4, 145.6, 197.6]
p50_latency = [76.0, 76.0, 80.8, 112.8]

# Configuração visual "HPC/NVIDIA Style"
plt.style.use('dark_background')
plt.figure(figsize=(12, 7))

# Plotando as linhas
plt.plot(req_per_sec, p99_latency, marker='o', markersize=8, color='#00ff00', label='P99 Latency (Tail)', linewidth=3)
plt.plot(req_per_sec, p50_latency, marker='s', markersize=8, color='#00bfff', label='P50 Latency (Median)', linewidth=2, linestyle='--')

# Destaque para o Ponto de Inflexão (Saturação)
plt.axvline(x=2500, color='red', linestyle=':', alpha=0.7)
plt.text(2505, 170, 'Critical Saturation (2.5k)', color='red', fontweight='bold')

# Estética do gráfico
plt.title('Mini-LLM Serving Saturation Analysis\n(Stories110M on RTX 2070)', fontsize=16, color='white', pad=20)
plt.xlabel('Load (Poisson Requests per Second)', fontsize=12)
plt.ylabel('End-to-End Latency (ms)', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.2)
plt.legend(fontsize=12)

# Salva
plt.tight_layout()
plt.savefig('latency_scaling_curve.png', dpi=300)
print("✅ Gráfico 'latency_scaling_curve.png' gerado com sucesso!")
