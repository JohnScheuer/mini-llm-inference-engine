import matplotlib.pyplot as plt
import numpy as np

# --- DADOS REAIS EXTRAÍDOS DOS SEUS BENCHMARKS ---
load_req_s = [10, 30, 60, 100, 500, 3000]
throughput_tok_s = [1050, 3200, 6125, 6125, 6125, 6125]
latency_p99_ms = [134.4, 134.4, 150.0, 10000.0, 35000.0, 69900.0]

plt.style.use('dark_background')
fig, ax1 = plt.subplots(figsize=(14, 8), dpi=200)

# 1. Plot de Throughput (Área Verde)
color_tps = '#00ff41' # Verde Matrix
ax1.set_xlabel('Input Load (Poisson Requests/sec)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Throughput (tokens/s)', color=color_tps, fontsize=12, fontweight='bold')
ax1.fill_between(load_req_s, throughput_tok_s, color=color_tps, alpha=0.2)
ax1.plot(load_req_s, throughput_tok_s, color=color_tps, linewidth=4, marker='o', markersize=8, label='Throughput')
ax1.tick_params(axis='y', labelcolor=color_tps)
ax1.set_xscale('log') # Escala logarítmica para ver a amplitude da carga

# 2. Plot de Latência P99 (Linha Vermelha)
ax2 = ax1.twinx()
color_lat = '#ff3333' # Vermelho Alerta
ax2.set_ylabel('Tail Latency P99 (ms)', color=color_lat, fontsize=12, fontweight='bold')
ax2.plot(load_req_s, latency_p99_ms, color=color_lat, linewidth=3, linestyle='--', marker='x', markersize=10, label='P99 Latency')
ax2.tick_params(axis='y', labelcolor=color_lat)
ax2.set_yscale('log') # Escala logarítmica para a latência que explode

# 3. ANOTAÇÕES DE ENGENHARIA
# Ponto de Saturação (O Joelho)
ax1.annotate('SATURATION KNEE\n6.1k tok/s @ 60 req/s', 
             xy=(60, 6125), xytext=(15, 5000),
             arrowprops=dict(facecolor='white', shrink=0.05),
             bbox=dict(boxstyle="round,pad=0.5", fc="#004400", ec="white", lw=1),
             fontsize=10, color='white', fontweight='bold')

# Zona de Queuing
plt.axvspan(60, 3000, color='red', alpha=0.1)
plt.text(150, 200, 'QUEUING DELAY ZONE\n(Hardware Saturated)', color='#ff3333', fontsize=12, fontweight='bold')

# Títulos
plt.title('Inference Engine Scalability: Throughput vs. Tail Latency\n(TinyLlama 1.1B • Full W8A8 • RTX 2070)', 
          fontsize=18, pad=30, fontweight='bold')

fig.tight_layout()
plt.savefig('throughput_latency_knee.png')
print("\n[OK] Gráfico 'throughput_latency_knee.png' gerado com sucesso!")
