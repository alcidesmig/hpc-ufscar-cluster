#!/bin/bash
sif_image="build/Container.simg"
sudo singularity build ${sif_image} images/Singularity.def 
scp ${sif_image} clufscar:~u760479/.
scp run_job.sh clufscar:~u760479/.
ssh clufscar # srun run_job.sh"  # need tty
