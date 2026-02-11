# Cluster commands - very short and essential guide
Tutorial for how to use python code: [Python example from Area Science Park](https://orfeo-doc.areasciencepark.it/examples/jupyterlab/)

## Help commands:
```bash
# SLURM command help
sbatch --help
srun --help
squeue --help

# Manual pages
man sbatch
man srun
man squeue

# Module help
module help python/3.11
```

## Available partitions

```bash
# View your account associations, allowed partitions and how much I can use them
sacctmgr list associations Users=$(whoami) format=Account,User,Partition,Priority,MaxJobs,MaxSubmit,MaxWall

# Show account details
sacctmgr show user $(whoami)
```

| Partition | CPUs | Memory | GPUs | Notes |
|-----------|------|--------|------|-------|
| **EPYC** | 128 cores (AMD EPYC) | 512GB | None | CPU-only computing |
| **THIN** | Varies | Varies | None | Standard nodes |
| **GPU** | 24 cores | 256GB | 2x V100 (32GB) | GPU computing |
| **GENOA** | AMD EPYC 9374F | 512GB | None | Latest AMD processors |


## Node States
- **idle**: Available for immediate use
- **alloc**: Fully allocated, all resources in use
- **mix**: Partially allocated, some resources available
- **down**: Offline for maintenance
- **drain**: Being prepared for maintenance


## Basic Information Commands

```bash
# Show partition summary
sinfo

# Detailed partition information
sinfo -o "%.12P %.5a %.10l %.6D %.6t %.10N"

# Detailed node information with resources
sinfo -N --format="%.15N %.6D %.10P %.11T %.4c %.10z %.8m %.10e %.9O %.15C"

# Show specific partition details
scontrol show partition GPU

# Show all partitions configuration
scontrol show partitions
```

## SBATCH Script Options

```bash
#!/bin/bash
#SBATCH --job-name=myjob           # Job name
#SBATCH --output=job_%j.out        # Output file (%j = job ID)
#SBATCH --error=job_%j.err         # Error file
#SBATCH --partition=gpu            # Partition
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=16        # CPUs per task
#SBATCH --mem=64G                 # Memory per node
#SBATCH --time=02:00:00           # Time limit (HH:MM:SS)
#SBATCH --account=your_account    # Account to charge
#SBATCH --mail-type=ALL           # Email notifications
#SBATCH --mail-user=your@email    # Email address
#SBATCH --gres=gpu:V100:1         # GPU resources
#SBATCH --chdir=/path/to/workdir  # Working directory
#SBATCH --array=1-10              # Job array
#SBATCH --exclusive               # Exclusive node access
#SBATCH --constraint=feature      # Node features required
```

## Job Control (and debug)

```bash
# To sbatch the job (define memory/compute node/time inside sbatch file)
bash <file>.sh

# Cancel a job
scancel 12345

# Cancel all your jobs
scancel -u $(whoami)

# Run job with debug output
sbatch --verbose script.sh

# Check why job is pending
squeue -j 12345 --long
scontrol show job 12345 | grep Reason

# Check actual resource usage after job
sacct -j 12345 --format=JobID,Elapsed,NCPUs,MaxRSS,MaxVMSize

# If I get end of line error because of saving my files in Windows:
sed -i 's/\r$//' <file>
```

## Queue Information

```bash
# View all jobs in queue
squeue

# View your jobs only
squeue -u $(whoami)

# Detailed queue information
squeue --long

# Show job start times estimates
squeue --start

# Show jobs for specific account
squeue -A <your_account>
```


## Module management
```bash
# List all available modules
module avail

# List available modules by category
module avail mpi
module avail python
module avail cuda
module avail compiler

# List loaded modules
module list

# Assure no modules
module purge
# Load necessary modules
module <?>

# Save current module environment
module save my_environment

# Restore saved environment
module restore my_environment

# List saved environments
module savelist
```

## Copy result/scripts from my computer to cluster
```bash
# To copy only the files I need from my folder to cluster (execute from my computer, not inside cluster)
scp scripts/... sbatch_scripts/... orfeo:~/Project/

# To copy results from cluster (execute from my computer, not inside cluster)
scp -r orfeo:~/Project/output/ ./
```

