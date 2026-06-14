#!/bin/bash
# Configurações de performance para o Ryzen 5 5600X
export OMP_NUM_THREADS=6
export OMP_PROC_BIND=true
export OMP_PLACES=cores

# Executa o motor
./mini-llm-engine