#!/bin/bash 
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling  
#################
#set a job name  
#SBATCH --job-name=deeplift
#################  
#a file for job output, you can check job progress
#SBATCH --output=Jobs/deeplift-%j.out
#################
# a file for errors from the job
#SBATCH --error=Jobs/deeplift-%j.err
#################
#time you think you need; default is one hour
#in minutes in this case
#SBATCH --time=48:00:00
#################
#SBATCH -p menon
#################
#quality of service; think of it as job priority
#SBATCH --qos=normal
#################
#SBATCH -G 1
#SBATCH -c 10
#################
#you could use --mem-per-cpu; they mean what we are calling cores
#################
#get emailed about job BEGIN, END, and FAIL
#SBATCH --mail-type=NONE
#SBATCH --mail-user=yuanzh@stanford.edu
#################
#now run normal batch commands
export PATH=SOFTWARE_DIR/miniconda/bin/:$PATH
source activate SOFTWARE_DIR/python_envs/dnn/

#Run the job
python step17_generate_DL_feature_attribution_multiple_cv.py
