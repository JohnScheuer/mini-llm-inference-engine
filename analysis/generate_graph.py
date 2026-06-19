import matplotlib.pyplot as plt
import numpy as np

# --- DADOS REAIS COLETADOS ---
load_input = [1000, 1500, 2000, 2300, 2500, 2600]
throughput = [16481.53, 21084.48, 22879.25, 25312.93, 26063.46, 26584.17]

# Configurações de Estilo
plt.style.use('dark_background')
fig, ax1 = plt.subplots(figsize=(12, 7), dpi=200)

# 1. Plot de Throughput (Eixo Esquerdo)
color_tps = '#00ff00' # Verde Neon
ax1.set_xlabel('Input Load (Poisson Requests/sec)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Token Throughput (tok/s)', color=color_tps, fontsize=12, fontweight='bold')
line1 = ax1.plot(load_input, throughput, marker='o', markersize=10, 
                color=color_tps, linewidth=4, label='Measured Throughput')
ax1.tick_params(axis='y', labelcolor=color_tps)
ax1.grid(True, which='both', linestyle='--', alpha=0.2)

# Linha de Limite de Hardware
ax1.axhline(y=26600, color='yellow', linestyle=':', alpha=0.6)
ax1.text(1050, 26800, 'RTX 2070 Saturation Point (~26.6k)', color='yellow', fontsize=10, fontweight='bold')

# 2. Plot de Latência (Eixo Direito)
ax2 = ax1.twinx()
color_lat = '#ff00ff' # Rosa/Magenta Neon
ax2.set_ylabel('P99 Latency (ms)', color=color_lat, fontsize=12, fontweight='bold')
ax2.plot(load_input, [134.40]*6, color=color_lat, linestyle='--', linewidth=2, label='P99 Latency (SLA)')
ax2.set_ylim(0, 200)
ax2.tick_params(axis='y', labelcolor=color_lat)

plt.title('Mini-LLM Final Report: Throughput Saturation & SLA Stability\n(Stories110M @ RTX 2070)', 
          fontsize=16, pad=25, fontweight='bold')

lines = line1 + ax2.get_lines()
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower right', frameon=True, alpha=0.8)

plt.tight_layout()
plt.savefig('final_benchmark_graph.png')
print("Gráfico 'final_benchmark_graph.png' gerado com sucesso!")
