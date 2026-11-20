#!/usr/bin/env bash

cnames=(Random)

nets=(all_to_all)

configs=(
  # 扩展 QPU 数量：每个 QPU 40 qubits
  # "80 2 40"
  # "120 3 40"
  # "160 4 40"
  # "200 5 40"
  # "240 6 40"
  # "280 7 40"

  # 扩展 qubit 数量：固定使用 3 个 QPU
  # "120 3 40"
  "150 3 50"
  # "180 3 60"
  # "210 3 70"
  # "240 3 80"
  # "270 3 90"
  # "300 3 100"
)

for cname in "${cnames[@]}"; do
  for net in "${nets[@]}"; do
    for config in "${configs[@]}"; do
      read -r nq core cap <<< "$config"
      python main.py -cname "$cname" -nq "$nq" -core "$core" -cap "$cap" -net "$net" -ocsv scalability.csv
    done
  done
done
