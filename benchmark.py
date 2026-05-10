import time
import tracemalloc
import sys
from tabulate import tabulate
from src.main import RobotSwarm

def run_benchmark():
    # Number of robots to test
    num_robots_list = [1000, 2500, 5000, 10000]
    
    # Collision radius
    collision_radius = 5.0
    
    results = []
    
    print("=====================================================")
    print("  SwarmDB Kinetic Join Benchmark Suite")
    print("=====================================================")
    print("Target Architecture: CPU Baseline (Pre-CUDA)")
    print("Complexity: O(N^2)")
    print("-----------------------------------------------------\n")
    
    for n in num_robots_list:
        print(f"[*] Benchmarking with N = {n} robots...")
        swarm = RobotSwarm(num_robots=n)
        
        # Start tracing memory allocations
        tracemalloc.start()
        
        # Start execution timer
        start_time = time.perf_counter()
        
        # Execute the Kinetic Join
        collisions = swarm.find_collisions(radius=collision_radius)
        
        # Stop timer
        end_time = time.perf_counter()
        
        # Capture memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        exec_time = end_time - start_time
        peak_mb = peak / (1024 * 1024)
        num_collisions = len(collisions)
        
        results.append([
            n, 
            f"{exec_time:.4f} s", 
            f"{peak_mb:.2f} MB", 
            num_collisions
        ])

    print("\n=== Benchmark Results ===")
    headers = ["Number of Robots (N)", "Execution Time", "Peak Memory Allocation", "Collisions Found"]
    print(tabulate(results, headers=headers, tablefmt="github"))
    print("\nNote: Peak memory measures the allocation overhead during distance calculation.")

if __name__ == "__main__":
    run_benchmark()
