# SwarmDB: Massively Parallel Spatial Indexing Engine for Autonomous Systems

**SwarmDB** is a high-performance spatial database engine conceptualized for the NVIDIA CINECA hackathon. Its primary goal is to manage and query real-time telemetry data for autonomous robotic swarms using GPU acceleration.

This repository currently implements the **CPU Baseline** version of the engine in Python, heavily relying on NumPy's vectorized operations to demonstrate the mathematical logic and the inherent computational bottlenecks. 

## Overview

In autonomous swarm robotics, verifying the proximity of agents in real-time is vital to prevent collisions and coordinate formations. SwarmDB simulates a dense environment of 10,000 active robots navigating a 3D coordinate space. 

The core operation of this database is the **Kinetic Join**: a continuous query that identifies all pairs of robots falling within a specific collision radius. 

## Current Architecture

The CPU-bound implementation utilizes the following structure:
- `src/main.py`: The `RobotSwarm` class acts as the API and state manager for the swarm, maintaining $(x, y, z)$ coordinates and simulating movement.
- `src/kernel.py`: The computational engine resolving the Kinetic Join using matrix expansion: $(A-B)^2 = A^2 + B^2 - 2A \cdot B$.
- `benchmark.py`: An evaluation suite to measure throughput and memory overhead of the engine across varying swarm sizes.

## Current Bottlenecks

Calculating pairwise distances for a swarm requires comparing every robot against every other robot, leading to an **$O(N^2)$ time complexity**. 

For $N = 10,000$, a single Kinetic Join necessitates generating and processing an $N \times N$ matrix containing $100,000,000$ distance values. 
- **Compute Limitation**: On a standard CPU, resolving this matrix takes noticeable time (milliseconds to seconds), rendering real-time execution across multiple ticks unviable.
- **Memory Overhead**: Allocating memory for continuous evaluation of dense $N \times N$ structures acts as a significant memory bottleneck, eventually causing out-of-memory faults as $N$ scales to larger commercial swarm sizes (e.g., $100,000+$).

## Optimization Goals for the Leonardo Cluster

This project is tailored for deployment and optimization on the **Leonardo supercomputer cluster** leveraging its dense NVIDIA GPU accelerators. The next phases of this project involve transitioning the CPU baseline to a high-throughput CUDA kernel:

1. **Massive Parallelism**: Porting the Kinetic Join to PyCUDA or raw CUDA C++, assigning individual threads to compute specific coordinate differences rather than computing the entire matrix in local CPU memory.
2. **Shared Memory Optimization**: Storing coordinate blocks in GPU Shared Memory to minimize global memory bandwidth saturation during the $A \cdot B$ matrix dot products.
3. **Spatial Indexing (Phase 2)**: Transitioning from a brute-force $O(N^2)$ dense matrix calculation to a localized bounding-volume hierarchy (e.g., Octree or GPU Hash Grid) to reduce theoretical complexity closer to $O(N \log N)$.

## Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Benchmark Suite
To measure the $O(N^2)$ baseline metrics on your current CPU hardware:
```bash
python benchmark.py
```

### 3. Example Output
```text
=====================================================
  SwarmDB Kinetic Join Benchmark Suite
=====================================================
Target Architecture: CPU Baseline (Pre-CUDA)
Complexity: O(N^2)
-----------------------------------------------------

[*] Benchmarking with N = 1000 robots...
[*] Benchmarking benchmarking with N = 2500 robots...
[*] Benchmarking benchmarking with N = 5000 robots...
[*] Benchmarking benchmarking with N = 10000 robots...

=== Benchmark Results ===
|   Number of Robots (N) | Execution Time   | Peak Memory Allocation   |   Collisions Found |
|------------------------|------------------|--------------------------|--------------------|
|                   1000 | 0.0075 s         | 7.63 MB                  |                 14 |
|                   2500 | 0.0382 s         | 47.68 MB                 |                 65 |
|                   5000 | 0.1534 s         | 190.73 MB                |                287 |
|                  10000 | 0.7041 s         | 762.94 MB                |               1150 |
```
*(Results will vary based on hardware)*
