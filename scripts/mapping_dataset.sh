#!/bin/bash

# This script runs the render_data.py script N times with different instance IDs and seeds.

N=50

echo "Running the script for $N iterations..."

for i in $(seq 0 $(($N - 1))); do
  echo "=================================================="
  echo "Starting iteration $i with instance_id=$i and seed=$i"
  echo "=================================================="
  python dataset/render_data.py --activity_instance_id $i --seed $i
  echo "--------------------------------------------------"
  echo "Finished iteration $i."
  echo "--------------------------------------------------"
done

echo "All $N iterations completed."
