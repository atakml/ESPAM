import os
import subprocess
import psutil
import torch
import time

def check_resources():
    # Example resource check: CPU usage, available memory, and CUDA availability
    cpu_usage = psutil.cpu_percent(interval=1)
    available_memory = psutil.virtual_memory().available / (1024 ** 2)  # in MB
    print(f"CPU Usage: {cpu_usage}%, Available Memory: {available_memory} MB")
    
    # Check for CUDA availability and free memory
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            free_cuda_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2) - torch.cuda.memory_allocated(0) / (1024 ** 2)  # in MB
            print(f"CUDA Available: {cuda_available}, Free CUDA Memory: {free_cuda_memory} MB")
        else:
            free_cuda_memory = 0
            print("CUDA not available")
    except ImportError:
        print("PyTorch not installed, unable to check CUDA")
        cuda_available = False
        free_cuda_memory = 0

    # Define thresholds
    if cpu_usage < 80 and available_memory > 500 and cuda_available and free_cuda_memory > 5120:  # 10 GB
        return True
    return False

def run_gstar(protein, seed):
    command = f"python3 gstar_ba.py --protein {protein} --seed {seed}"
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}: {command}")

if __name__ == "__main__":
    # Wait until there is no gstar_ba.py process running
    while any("gstar_ba.py" in p.info['cmdline'] for p in psutil.process_iter(attrs=['cmdline'])):
        print("gstar_ba.py is running. Waiting for it to finish...")
        time.sleep(60)
    
    for protein in ["OR1A1", "OR5K1", "OR7D4", "OR51E2"]:
        for seed in range(5):
            if os.path.exists(f"GSTARX_res_{protein}_{seed}.pkl"):
                print(f"File GSTARX_res_{protein}_{seed}.pkl already exists. Skipping...")
                continue
            print(f"Checking resources for protein={protein}, seed={seed}...")
            if check_resources():
                subprocess.Popen(
                    ["python3", "gstar_ba.py", "--protein", protein, "--seed", str(seed)]
                )
                time.sleep(60)
            else:
                print("Insufficient resources")
                while not check_resources():
                    print("Resources unavailable. Retrying in 10 seconds...")
                    time.sleep(19)