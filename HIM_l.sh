#!/bin/bash
source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate himenv3
torchrun --nproc_per_node=4 HIM_l.py
