import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tracemalloc
from allreduce import butterfly_allreduce, ring_allreduce

def peak_memory(func, *args, **kwargs):
    tracemalloc.start()
    start_time = time.time()
    func(*args, **kwargs)
    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak

def worker(algorithm, rank, world_size, vector_size, results):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    torch.manual_seed(42)
    tensor = torch.randn(vector_size, dtype=torch.float32)
    expected = tensor.clone()
    dist.all_reduce(expected)
    expected /= world_size

    torch.manual_seed(42)
    tensor = torch.randn(vector_size, dtype=torch.float32)
    dist.barrier()
    
    if algorithm == 'butterfly':
        t, m = peak_memory(butterfly_allreduce, tensor, rank, world_size)
    else:
        t, m = peak_memory(ring_allreduce, tensor, rank, world_size)

    acc = float(torch.mean(torch.abs(tensor - expected)))
    if rank == 0:
        results.put((algorithm, world_size, vector_size, t, m, acc))
    
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    print("="*80)
    print("BUTTERFLY vs RING ALL-REDUCE (Measured Peak Memory)")
    print("="*80)
    
    configs = [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32)]
    
    for workers, vec_size in configs:
        ctx = mp.get_context('spawn')
        for algorithm in ['butterfly', 'ring']:
            results = ctx.Queue()
            processes = []
            for rank in range(workers):
                p = ctx.Process(target=worker, args=(algorithm, rank, workers, vec_size, results))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            
            while not results.empty():
                algo, w, v, t, m, acc = results.get()
                print(f"Workers {w}, Vector {v} | {algo:9} | Time: {t:.6f}s | Peak Memory: {m/1024:8.1f}KB | Accuracy diff: {acc:.6e}")
