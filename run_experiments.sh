#!/bin/bash

# Train HiM models
echo "Training HiM with Lorentz manifold..."
python HIM_l.py

echo "Training HiM with Poincaré manifold..."
python HIM_p.py

echo "Training complete! Results are stored in experiments/ directory"
