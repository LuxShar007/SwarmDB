import time
import tracemalloc
import sys
from tabulate import tabulate
from src.main import RobotSwarm

def run_benchmark():
    # Number of robots to test
    num_robots_list = [1000, 2500, 5000, 10000]
    large_num_robots_list = [25000, 50000]
    
    # Collision radius
    collision_radius = 5.0
    
    results = []
    
    print("=====================================================")
    print("  SwarmDB Kinetic Join Benchmark Suite")
    print("=====================================================")
    print("Comparing: CPU Baseline (O(N^2)) vs CPU Optimized (cKDTree O(N log N))")
    print("-----------------------------------------------------\n")
    
    # 1. Benchmark Baseline
    for n in num_robots_list:
        print(f"[*] Benchmarking Baseline with N = {n} robots...")
        swarm = RobotSwarm(num_robots=n)
        
        tracemalloc.start()
        start_time = time.perf_counter()
        collisions = swarm.find_collisions(radius=collision_radius, method="baseline")
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        exec_time = end_time - start_time
        peak_mb = peak / (1024 * 1024)
        num_collisions = len(collisions)
        
        results.append([
            n, 
            "Baseline",
            f"{exec_time:.4f} s", 
            f"{peak_mb:.2f} MB", 
            num_collisions
        ])

    # 2. Benchmark Optimized
    for n in num_robots_list + large_num_robots_list:
        print(f"[*] Benchmarking Optimized with N = {n} robots...")
        swarm = RobotSwarm(num_robots=n)
        
        tracemalloc.start()
        start_time = time.perf_counter()
        collisions = swarm.find_collisions(radius=collision_radius, method="optimized")
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        exec_time = end_time - start_time
        peak_mb = peak / (1024 * 1024)
        num_collisions = len(collisions)
        
        results.append([
            n, 
            "Optimized",
            f"{exec_time:.4f} s", 
            f"{peak_mb:.2f} MB", 
            num_collisions
        ])

    # 3. Benchmark CUDA (if available)
    cuda_available = False
    try:
        import cupy  # type: ignore
        cuda_available = True
    except ImportError:
        try:
            import pycuda  # type: ignore
            cuda_available = True
        except ImportError:
            pass

    if cuda_available:
        print("\n[*] CUDA is available! Benchmarking CUDA GPU method...")
        for n in num_robots_list + large_num_robots_list:
            try:
                print(f"[*] Benchmarking CUDA with N = {n} robots...")
                swarm = RobotSwarm(num_robots=n)
                
                tracemalloc.start()
                start_time = time.perf_counter()
                collisions = swarm.find_collisions(radius=collision_radius, method="cuda")
                end_time = time.perf_counter()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                exec_time = end_time - start_time
                peak_mb = peak / (1024 * 1024)
                num_collisions = len(collisions)
                
                results.append([
                    n, 
                    "CUDA GPU (Shared)",
                    f"{exec_time:.4f} s", 
                    f"{peak_mb:.2f} MB", 
                    num_collisions
                ])
            except Exception as e:
                print(f"[!] CUDA benchmark failed for N = {n}: {e}")
                tracemalloc.stop()
    else:
        print("\n[!] CUDA is not available. Skipping GPU benchmarks.")

    print("\n=== Benchmark Results ===")
    headers = ["Robots (N)", "Method", "Execution Time", "Peak Memory Allocation", "Collisions Found"]
    print(tabulate(results, headers=headers, tablefmt="github"))
    print("\nNote: Peak memory measures the allocation overhead during distance calculation.")

if __name__ == "__main__":
    run_benchmark()


