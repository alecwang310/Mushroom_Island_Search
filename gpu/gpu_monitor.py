"""GPU monitor: polls nvidia-smi during the hunt to log utilization."""
import subprocess, time, sys

def monitor(interval=1.0):
    """Print GPU stats every `interval` seconds until stdin closes."""
    cmd = ['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,temperature.gpu,clocks.sm,clocks.mem',
           '--format=csv,noheader']
    try:
        while True:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            print(result.stdout.strip(), flush=True)
            time.sleep(interval)
    except (KeyboardInterrupt, EOFError):
        pass

if __name__ == '__main__':
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    monitor(interval)
