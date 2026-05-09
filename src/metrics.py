
import json, torch, time, os
class MetricsTracker:
    def __init__(self, strategy):
        self.strategy = strategy
        self.history = []
    def record(self, step, loss, start_time):
        self.history.append({
            "step": step, "loss": float(loss),
            "gpu_memory_mb": torch.cuda.max_memory_allocated() / 1e6,
            "step_time_s": time.time() - start_time
        })
    def save(self):
        path = f"outputs/{self.strategy}/metrics.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f: json.dump(self.history, f)
