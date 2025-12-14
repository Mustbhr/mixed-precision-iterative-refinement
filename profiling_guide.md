# Profiling Guide for IBEX (Nsight)

This guide walks you through profiling your mixed-precision solver on the KAUST IBEX cluster.

## 1. The Profiling Script
I have created a new SLURM script for you: `profile_job.slurm`.
It does two things:
1.  **Nsight Systems (`nsys`)**: Captures the timeline (CPU/GPU overlap).
2.  **Nsight Compute (`ncu`)**: Captures kernel metrics (Tensor Core utilization, Memory Bandwidth).

## 2. Running on IBEX
Login to IBEX and submit the job:
```bash
# 1. Login
ssh glogin.ibex.kaust.edu.sa

# 2. Go to project directory
cd /path/to/CS380-Project

# 3. Submit the profiling job
sbatch profile_job.slurm
```
*Note: The script requests an A100 GPU and 30 minutes of time.*

## 3. Retrieving Results
Once the job finishes (check with `squeue -u <username>`), you will see two new files ending in `.nsys-rep` and `.ncu-rep`. Copy them to your local laptop:

```bash
# On your local laptop:
scp "ibex:/path/to/CS380-Project/report_timeline_*.nsys-rep" .
scp "ibex:/path/to/CS380-Project/report_kernels_*.ncu-rep" .
```

## 4. Viewing Results
You need to install the **NVIDIA Nsight Systems** and **Nsight Compute** viewers on your local machine (Windows/Mac/Linux).

### Analyzing Timeline (`.nsys-rep`)
1.  Open **Nsight Systems**.
2.  Open the file.
3.  Look for the **CUDA HW** row. You want to see the blue blocks (Kernels) tightly packed with no gaps (which would indicate CPU slowness).

### Analyzing Kernels (`.ncu-rep`)
1.  Open **Nsight Compute**.
2.  Open the file.
3.  Go to the **"Speed of Light"** section.
4.  **Roofline Chart**: You want your dot to be near the top (Compute Bound) or top-right.
5.  **Tensor Core**: Check "Compute Workload Analysis" > "Pipeline Utilization". You should see **Tensor Cores** highly active.
