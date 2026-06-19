import matplotlib.pyplot as plt
import numpy as np

# Configurações do Hardware (RTX 2070)
peak_compute = 50000  # 50 TFLOPS em GFLOPS
bandwidth = 448       # 448 GB/s

# Dados reais do seu motor
intensity_110m = 1.0  # FP16 Batch 1
perf_110m = 450       # GFLOPS aprox

intensity_1b = 2.0    # INT8 (355 tok/s)
perf_1b = 782.12      # GFLOPS real

# Criar a linha do Roofline
x_ridge = peak_compute / bandwidth
x = np.logspace(-1, 3, 100)
y = np.minimum(peak_compute, bandwidth * x)

plt.style.use('dark_background')
plt.figure(figsize=(12, 7), dpi=200)

# Plot da curva Roofline
plt.loglog(x, y, label='RTX 2070 Theoretical Limit', color='white', linewidth=2, alpha=0.5)

# Plot dos seus pontos
plt.scatter(intensity_110m, perf_110m, color='cyan', s=150, edgecolors='white', zorder=5, label='Stories 110M (FP16)')
plt.scatter(intensity_1b, perf_1b, color='#ff00ff', s=200, marker='D', edgecolors='white', zorder=5, label='TinyLlama 1.1B (INT8 - 87% Eff)')

# Anotações
plt.annotate(f'TinyLlama 1.1B\n{perf_1b:.1f} GFLOPS', 
             xy=(intensity_1b, perf_1b), xytext=(intensity_1b*1.5, perf_1b*0.5),
             arrowprops=dict(arrowstyle='->', color='magenta'))

plt.axvline(x=x_ridge, color='yellow', linestyle='--', alpha=0.3)
plt.text(x_ridge*1.1, 10, f'Ridge Point ({x_ridge:.1f} FLOP/B)', color='yellow', fontsize=10)

plt.title('Roofline Analysis: 110M vs. 1.1B Scaling\n(RTX 2070 High-Performance Engine)', fontsize=16, pad=20)
plt.xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=12)
plt.ylabel('Performance (GFLOPS)', fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.1)
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('roofline_final.png')
print("Gráfico 'roofline_final.png' gerado com sucesso!")
