#!/usr/bin/env bash

cnames=(
  QFTAdder
  PQC
)

for cname in "${cnames[@]}"; do
  python main.py \
    -gconf config.navi_ablation_no_max_match.json \
    -cname "$cname" \
    -nq 80 \
    -core 4 \
    -cap 20 \
    -net chain \
    -ocsv navi_ablation_no_max_match.csv
done
