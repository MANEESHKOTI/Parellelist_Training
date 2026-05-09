import json, os
import matplotlib.pyplot as plt

paths = {"DDP": "outputs/ddp/metrics.json", "ZeRO-1": "outputs/zero1/metrics.json", "ZeRO-2": "outputs/zero2/metrics.json"}
data = {}
for s, p in paths.items():
    if os.path.exists(p):
        with open(p, 'r') as f: j = json.load(f)
        data[s] = {"steps": [e["step"] for e in j], "mem": [e["gpu_memory_mb"] for e in j], "time": [e["step_time_s"] for e in j]}

os.makedirs("outputs/plots", exist_ok=True)

# Memory Plot
plt.figure(figsize=(8, 5))
for s, d in data.items(): plt.plot(d["steps"], d["mem"], label=s, marker='o')
plt.title('Memory Usage Comparison'); plt.xlabel('Step'); plt.ylabel('GPU Memory MB'); plt.legend(); plt.grid()
plt.savefig("outputs/plots/gpu_memory_comparison.png")

# Throughput Plot
plt.figure(figsize=(8, 5))
for s, d in data.items(): plt.plot(d["steps"], d["time"], label=s, marker='s')
plt.title('Throughput Comparison (Step Time)'); plt.xlabel('Step'); plt.ylabel('Step Time Seconds'); plt.legend(); plt.grid()
plt.savefig("outputs/plots/throughput_comparison.png")
