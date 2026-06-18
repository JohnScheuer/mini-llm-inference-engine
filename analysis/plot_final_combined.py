import matplotlib.pyplot as plt

# -------------------------------
# Dados coletados
# -------------------------------

req_per_sec = [2300, 2500, 2600]
throughput = [29974, 30000, 32000]  # média aproximada
p99_latency = [134.4, 145.6, 197.6]

# -------------------------------
# Criar gráfico combinado
# -------------------------------

fig, ax1 = plt.subplots(figsize=(10,6))

# Throughput
color = 'tab:blue'
ax1.set_xlabel("Arrival Rate (req/s)")
ax1.set_ylabel("Throughput (tok/s)", color=color)
ax1.plot(req_per_sec, throughput, marker='o', color=color, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)

# Segundo eixo para latência
ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel("P99 Latency (ms)", color=color)
ax2.plot(req_per_sec, p99_latency, marker='s', color=color, linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Throughput & Tail Latency vs Arrival Rate\nStories110M – RTX 2070")

fig.tight_layout()
plt.grid(True)

plt.savefig("final_scaling_combined.png", dpi=300)
plt.show()

print("✅ final_scaling_combined.png gerado com sucesso!")
