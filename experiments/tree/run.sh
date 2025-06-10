#!/bin/bash
#SBATCH --array=1-300%5
#SBATCH --job-name=sparse4pinn
#SBATCH --nodes=1                # nombre de noeuds
#SBATCH --ntasks=1               # nombre total de tâches sur tous les nœuds
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=4G
#SBATCH --output=hs_slurm/dcv_hist/out/slurm-%A_%a.txt
#SBATCH --error=hs_slurm/dcv_hist/err/slurm-%A_%a.txt
#SBATCH --mail-type=ALL
#SBATCH --requeue
#SBATCH --gres=gpu:1

# export TMPDIR=/scratch/<project>/tmp

DIR=$1
PARAMS_OFFSET=$2

PARAMS_FILE="${DIR}/params"
RUNLOG_FILE="${DIR}/runlog"
BATCH_HIST="batch.txt"

if [ -z "$PARAMS_OFFSET" ]
then
    PARAMS_OFFSET=0
fi

if [ ! -d "$DIR" -o ! -f "$PARAMS_FILE" ]
then
    echo "Usage: $0 DIR [PARAMS_OFFSET]"
    echo "where DIR is a directory containing a file 'params' with the parameters."
    exit 1
fi

PARAMS_ID=$(( $SLURM_ARRAY_TASK_ID + $PARAMS_OFFSET ))
JOB_NAME="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "$PARAMS_ID|$JOB_NAME|$SLURM_SUBMIT_DIR" >> $RUNLOG_FILE

PARAMS=$(tail -n +${PARAMS_ID} ${PARAMS_FILE} | head -n 1)

CMD=$"srun python -m experiments.tree.run ${PARAMS}"

echo "start"
ml python/3.12
ml cuda/12.4
echo "$PARAMS"
source ../env/bin/activate # change to your virtual environment
export PYTHONPATH=$PWD 
echo "$PARAMS_ID|$JOB_NAME|$SLURM_SUBMIT_DIR|$CMD" >> $BATCH_HIST
$CMD
deactivate
echo "end"