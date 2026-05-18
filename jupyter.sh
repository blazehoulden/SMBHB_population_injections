#!/bin/bash
#SBATCH --nodes 1
#SBATCH --time 4:00:00
#SBATCH --job-name jupyter-analysis
#SBATCH --output jupyter-log-%J.txt
## get tunneling info
XDG_RUNTIME_DIR=""
ipnport=$(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)
## print tunneling instructions to jupyter-log-{jobid}.txt
echo -e "
Copy/Paste this in your local terminal to ssh tunnel with remote
-----------------------------------------------------------------
ssh -N -L $ipnport:$ipnip:$ipnport user@host
-----------------------------------------------------------------
Then open a browser on your local machine to the following address
------------------------------------------------------------------
localhost:$ipnport (prefix w/ https:// if using password)
------------------------------------------------------------------
"
module load mamba
mamba activate smbhb312
cd /fred/oz005/users/bhoulden/SMBHB_population_injections/
## start an ipcluster instance and launch jupyter server
jupyter-notebook --no-browser --port=$ipnport --ip=$ipnip
