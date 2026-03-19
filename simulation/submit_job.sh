#!/bin/bash
#SBATCH --job-name=foundation_stereo
#SBATCH --output=stereo_out.log
#SBATCH --error=stereo_err.log
#SBATCH --time=04:00:00              # Request 4 hours (adjust as needed)
#SBATCH --partition=tron             # Standard UMIACS partition
#SBATCH --account=nexus              # Use 'nexus' or your specific group account
#SBATCH --qos=default                # Use 'default' or 'medium'
#SBATCH --mem=32G                    # Request 64GB of system RAM
#SBATCH --gres=gpu:rtxa6000:1        # Request 1 RTX A6000 GPU (48GB VRAM)

# Load necessary modules (adjust based on your environment)
module load python/3.11
# If you use Conda, activate your environment here:
# source /homes/youruser/anaconda3/bin/activate foundation_stereo

# Run your script
python pipeline.py 


# sbatch submit_job.sh
# squeue -u akilax0
# scancel JOBID
