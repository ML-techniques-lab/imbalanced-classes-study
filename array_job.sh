#!/bin/bash
step=$((100 / $SLURM_ARRAY_TASK_MAX))
start=$(( ((($SLURM_ARRAY_TASK_ID - 1) * step) + 1) ))
end=$(( $SLURM_ARRAY_TASK_ID * step ))

source venv/bin/activate
python experiment.py all ${1} $start:$end "results_2/${1}_${SLURM_ARRAY_TASK_ID}.csv"
