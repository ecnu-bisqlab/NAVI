cnames=(VQE)

nets=(mesh_grid)
configs=(
  "80 4 20"
)

for cname in "${cnames[@]}"; do
  for net in "${nets[@]}"; do
    for config in "${configs[@]}"; do
      read -r nq core cap <<< "$config"
      python main.py -cname "$cname" -nq "$nq" -core "$core" -cap "$cap" -net "$net"
    done
  done
done
