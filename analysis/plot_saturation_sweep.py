import matplotlib.pyplot as plt

# Dados consolidados dos seus benchmarks de 20.000 requisições
req_per_sec = [1000, 1500, 1800, 2000, 2300, 2500, 2600]
throughput = [19418, 28974, 30002, 31359, 30283, 32482, 32404]

plt.style.use('dark_background')
fig, ax1 = plt.subplots(figsize=(12, 7))

# --- Eixo 1: Throughput (Barras Verdes) ---
color_thru = '#00ff00'
ax1.set_xlabel('Arrival Rate (Poisson Requests/sec)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Total Throughput (tokens/s)', color=color_thru, fontsize=12, fontweight='bold')
ax1.bar(req_per_sec, throughput, width=80, color=color_thru, alpha=0.3, label='System Throughput')
ax1.tick_params(axis='y', labelcolor=color_thru)
ax1.set_ylim(0, 40000)

# --- Eixo 2: Latência P99 (Linha Laranja) ---
ax2 = ax1.twinx()
color_p99 = '#ff9900'
ax2.set_ylabel('P99 Tail Latency (ms)', color=color_p99, fontsize=12, fontweight='bold')
# Usando dados de regime estável filtrados para visualização de SLA
ax2.plot(req_per_sec, [134, 134, 135, 135, 135, 145, 198], color=color_p99, marker='D', linewidth=4, markersize=10, label='P99 Stability')
ax2.tick_params(axis='y', labelcolor=color_p99)
ax2.set_ylim(0, 250)

# --- Anotação do Limite Físico ---
ax1.annotate('HARDWARE LIMIT\n~32,400 tok/s', xy=(2600, 32421), xytext=(1500, 35000),
             arrowprops=dict(facecolor='white', shrink=0.05), fontsize=12, fontweight='bold', color='white', 
             bbox=dict(boxstyle="round,pad=0.3", fc="red", ec="white", alpha=0.5))

plt.title('Mini-LLM Final Report: Throughput Saturation & SLA Stability\n(Stories110M @ RTX 2070)', fontsize=16, pad=20)
ax1.grid(True, which='both', linestyle='--', alpha=0.1)

fig.tight_layout()
plt.savefig('final_saturation_sweep.png', dpi=300)
print("✅ Gráfico 'final_saturation_sweep.png' gerado com sucesso!")
