
import os
import sys
import torch
import time
import numpy as np
from PIL import Image
from tqdm import tqdm

# Ensure root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.inference import InferenceEngine
from training import config

def benchmark_latency():
    print("Initializing Latency Benchmark...")
    
    # 1. Setup Devices
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
        
    results = {}
    
    # Create dummy input
    dummy_img = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE), color='red')
    
    for device in devices:
        print(f"\nBenchmarking on {device.upper()}...")
        engine = InferenceEngine(model_path=str(config.CHECKPOINT_DIR / "best_model.pth"), device=device)
        
        # Warmup
        print("Warming up...")
        for _ in range(10):
            engine.predict(dummy_img)
            
        # Benchmark Single Image (Batch Size 1)
        print("Benchmarking Single Image (Batch Size 1)...")
        times = []
        for _ in range(100):
            start = time.time()
            engine.predict(dummy_img)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000) # ms
            
        avg_ms = np.mean(times)
        fps = 1000 / avg_ms
        p99 = np.percentile(times, 99)
        
        results[f"{device}_bs1"] = {
            "avg_ms": avg_ms,
            "fps": fps,
            "p99_ms": p99
        }
        print(f"  Avg Latency: {avg_ms:.2f} ms")
        print(f"  FPS: {fps:.2f}")
        
        # Benchmark Batch Processing (Batch Size 32)
        # We need to access the model directly for batch inference as engine.predict is single image
        print("Benchmarking Batch (Batch Size 32)...")
        batch_size = 32
        
        # Create batch tensor
        img_tensor = engine.transform(dummy_img).unsqueeze(0)
        batch_tensor = torch.cat([img_tensor] * batch_size).to(engine.device)
        
        times = []
        with torch.no_grad():
            # Warmup batch
            for _ in range(5):
                engine.model(batch_tensor)
                
            for _ in range(20): # 20 batches of 32 = 640 images
                start = time.time()
                engine.model(batch_tensor)
                if device == 'cuda':
                    torch.cuda.synchronize()
                end = time.time()
                times.append((end - start) * 1000) # ms per batch
                
        avg_batch_ms = np.mean(times)
        avg_per_image_ms = avg_batch_ms / batch_size
        fps = 1000 / avg_per_image_ms
        
        results[f"{device}_bs32"] = {
            "avg_batch_ms": avg_batch_ms,
            "avg_per_image_ms": avg_per_image_ms,
            "fps": fps
        }
        print(f"  Avg Batch Latency: {avg_batch_ms:.2f} ms")
        print(f"  Per Image Latency: {avg_per_image_ms:.2f} ms")
        print(f"  FPS: {fps:.2f}")

    # Generate Report
    report_path = config.RESULTS_DIR / "benchmark_report.txt"
    with open(report_path, "w") as f:
        f.write("==================================================\n")
        f.write("ATN DEER AGE RECOGNITION - LATENCY BENCHMARK\n")
        f.write("==================================================\n\n")
        
        for key, metrics in results.items():
            device, bs = key.split('_')
            bs_label = "Batch Size 1" if bs == "bs1" else "Batch Size 32"
            
            f.write(f"Device: {device.upper()} | {bs_label}\n")
            f.write("-" * 40 + "\n")
            if bs == "bs1":
                f.write(f"  Avg Latency:  {metrics['avg_ms']:.2f} ms\n")
                f.write(f"  p99 Latency:  {metrics['p99_ms']:.2f} ms\n")
                f.write(f"  Throughput:   {metrics['fps']:.2f} FPS\n")
            else:
                f.write(f"  Batch Latency:{metrics['avg_batch_ms']:.2f} ms\n")
                f.write(f"  Per Image:    {metrics['avg_per_image_ms']:.2f} ms\n")
                f.write(f"  Throughput:   {metrics['fps']:.2f} FPS\n")
            f.write("\n")
            
    print(f"\nBenchmark report saved to {report_path}")

if __name__ == "__main__":
    benchmark_latency()
