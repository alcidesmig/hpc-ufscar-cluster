#!/bin/bash
#SBATCH -J task_alcides       # Job name
#SBATCH -o %j.out             # Name of stdout output file (%j expands to jobId)
#SBATCH -n 1
#SBATCH -t 01:30:00           # Run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=alcidesms@estudante.ufscar.br
#SBATCH --mail-type=ALL

local_sing=.                               
local_job="/scratch/job.${SLURM_JOB_ID}"    

function clean_job() {
  echo "GC..."
  rm -rf "${local_job}"
}
trap clean_job EXIT HUP INT TERM ERR

set -eE

umask 077

sing=Container.simg

mkdir -p scratch

echo "Running..."
singularity run \
   --bind=/scratch:/scratch \
   --bind=/var/spool/slurm:/var/spool/slurm \
   Container.simg
