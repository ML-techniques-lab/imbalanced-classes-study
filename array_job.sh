step=$((100 / $SLURM_ARRAY_TASK_MAX))
start=$(( ($SLURM_ARRAY_TASK_ID - 1) * step))
end=$(( $SLURM_ARRAY_TASK_ID * step ))

mod=$(( 100 % $SLURM_ARRAY_TASK_MAX ))
if [ $mod != 0 ] && [ $SLURM_ARRAY_TASK_ID == $SLURM_ARRAY_TASK_MAX]; then
  end=$(( $end + mod ))

python experiment.py all GNB $start:$end "GNB_${step}.csv"