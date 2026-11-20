#!/usr/bin/env bash

cnames=(
  BV
  SwapTest
  Permutation
  MCMT
  Comparator
  QFTAdder
  CuccaroAdder
  PQC
  IQP
  Random
  QV
  VQE
)

for cname in "${cnames[@]}"; do
  python main.py \
    -gconf config.navi_ablation.json \
    -cname "$cname" \
    -nq 80 \
    -core 4 \
    -cap 20 \
    -net chain \
    -ocsv navi_ablation.csv
done
