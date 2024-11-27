#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuh100
### -- set the job Name --
#BSUB -J CUDA_test
### -- ask for number of cores (default: 1) --
#BSUB -n 100
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 1:00
# request 4GB of system-memory
#BSUB -R "rusage[mem=4GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
##BSUB -B
### -- send notification at completion--
##BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --


# -- Load modules --
module load python3
nvidia-smi
module load cuda

# -- Activate the virtual environment --
source /zhome/32/9/137127/Master-Thesis/.venv/bin/activate

# Execute Python file
python test.py