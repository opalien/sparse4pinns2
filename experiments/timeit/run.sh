
CMD=$"srun --pty --partition=gpu --gres=gpu:1 python experiments/timeit/run.py"

echo "start"
ml python/3.12
ml cuda/12.4
source ../env/bin/activate # change to your virtual environment
export PYTHONPATH=$PWD 
echo "$PARAMS_ID|$JOB_NAME|$SLURM_SUBMIT_DIR|$CMD" >> $BATCH_HIST
$CMD
deactivate
echo "end"