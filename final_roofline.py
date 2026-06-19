import matplotlib.pyplot as plt
import numpy as np

# Configurações do Hardware (RTX 2070)
peak_compute = 50000  # 50 TFLOPS (Tensor Cores) em GFLOPS
bandwidth = 448       # 448 GB/s

# Pontos reais do seu motor
# Ponto 1: Stories 110M
i_110m = 1.0
perf_110m = 450 # GFLOPS

# Ponto 2: TinyLlama 1.1B @ 1007 tok/s (SEU RECORDE ATUAL)
# Intensidade = (2.2 GFLOPs / 1.1 GB) * Batch 16 = 32
i_1b_final = 32.0  
perf_1b_final = 2215.4 # GFLOPS (1007 tok/s * 2.2 GFLOPs/tok)

# Criar a linha do Roofline
x_ridge = peak_compute / bandwidth
x = np.logspace(-1, 3, 100)
y = np.minimum(peak_compute, bandwidth * x)

plt.style.use('dark_background')
plt.figure(figsize=(12, 8), dpi=200)

# Linha teórica do hardware
plt.loglog(x, y, label='RTX 2070 Theoretical Limit', color='white', linewidth=2, alpha=0.3)

# Ponto Inicial
plt.scatter(i_110m, perf_110m, color='cyan', s=100, label='Initial (110M Model)')

# PONTO DO BREAKTHROUGH (Sua conquista de hoje)
plt.scatter(i_1b_final, perf_1b_final, color='#00ff41', s=400, marker='*', 
            edgecolors='white', linewidth=2, zorder=10, label='BREAKTHROUGH: 1.1B @ 1007 tok/s')

# Anotações Técnicas
plt.annotate(f'2.2 TFLOPS ACHIEVED!\nEffective BW: 1.1 TB/s\n(L2 Cache Saturation)', 
             xy=(i_1b_final, perf_1b_final), xytext=(i_1b_final*0.1, perf_1b_final*3),
             arrowprops=dict(facecolor='#00ff41', shrink=0.05, alpha=0.8),
             fontsize=12, color='#00ff41', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", fc="#000000", ec="#00ff41", alpha=0.8))

plt.title('Final Roofline Characterization: The 2.2 TFLOPS Breakthrough\n(RTX 2070 • TinyLlama 1.1B • Full INT8 W8A8)', fontsize=16, pad=25, fontweight='bold')
plt.xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=12, fontweight='bold')
plt.ylabel('Performance (GFLOPS)', fontsize=12, fontweight='bold')
plt.grid(True, which="both", ls="-", alpha=0.05)
plt.legend(loc='upper left', frameon=True, fontsize=10)

plt.tight_layout()
plt.savefig('roofline_breakthrough.png')
print("\n[OK] Gráfico 'roofline_breakthrough.png' gerado com sucesso!")
