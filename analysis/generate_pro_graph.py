import matplotlib.pyplot as plt

# --- DADOS REAIS QUE VOCÊ OBTEVE ---
load_input = [1000, 1500, 2000, 2300, 2500, 2600]
throughput = [16481.53, 21084.48, 22879.25, 25312.93, 26063.46, 26584.17]
latency_p99 = [134.40, 134.40, 134.40, 134.40, 134.40, 134.40] # Seus dados mostraram estabilidade total

# Configurações de Estilo Dark/High-End
plt.style.use('dark_background')
fig, ax1 = plt.subplots(figsize=(14, 8), dpi=200)

# 1. BARRAS DE THROUGHPUT (Estilo Pro)
color_tps = '#00ff41' # Verde Matrix/Neon
bars = ax1.bar(load_input, throughput, width=80, color=color_tps, alpha=0.3, label='Throughput (tok/s)')
ax1.set_xlabel('Arrival Rate (Poisson Requests/sec)', fontsize=12, fontweight='bold', labelpad=15)
ax1.set_ylabel('Total Throughput (tokens/s)', color=color_tps, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_tps)
ax1.grid(True, which='both', linestyle='--', alpha=0.1)

# 2. LINHA DE LATÊNCIA P99 (Eixo Direito)
ax2 = ax1.twinx()
color_lat = '#ff9900' # Laranja Neon
ax2.plot(load_input, latency_p99, marker='D', markersize=10, color=color_lat, 
         linewidth=4, label='P99 Tail Latency (ms)', markerfacecolor=color_lat)
ax2.set_ylabel('P99 Tail Latency (ms)', color=color_lat, fontsize=12, fontweight='bold')
ax2.set_ylim(0, 250) # Escala para mostrar como o P99 está baixo e estável
ax2.tick_params(axis='y', labelcolor=color_lat)

# 3. ANOTAÇÕES DE ENGENHARIA
# Marcar o limite de hardware atingido
max_tps = max(throughput)
ax1.annotate(f'HARDWARE LIMIT\n~{max_tps/1000:.1f}k tok/s', 
             xy=(2600, max_tps), xytext=(1500, max_tps+4000),
             arrowprops=dict(facecolor='white', shrink=0.05),
             bbox=dict(boxstyle="round,pad=0.5", fc="#880000", ec="white", lw=1),
             fontsize=12, color='white', fontweight='bold')

# Títulos
plt.title('Mini-LLM Final Report: Throughput Saturation & SLA Stability\n(Stories110M @ RTX 2070)', 
          fontsize=18, pad=30, fontweight='bold')

plt.tight_layout()
plt.savefig('final_report_neon.png')
print("Gráfico 'final_report_neon.png' gerado com sucesso!")
