# SwarmDB: Massively Parallel Spatial Indexing Engine for Autonomous Systems

A specialized, local-first spatial database engine explicitly engineered to resolve high-concurrency proximity constraints and collision-avoidance telemetry loops for autonomous multi-agent workflows.

### 🌐 Live Visualizer & Production Deployment
[![Visualize through our Website](https://img.shields.io/badge/Visualize%20through%20our%20Website-00FF00?style=for-the-badge&logo=vercel&logoColor=black&labelColor=111111)](https://arxhaven.github.io/SwarmDB-Demo/)

### 🛠️ Core Languages & Computational Frameworks
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![C++ / CUDA C](https://img.shields.io/badge/CUDA_C%2B%2B-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

### 🤖 Agentic AI & Model Inference Engines
![NVIDIA NeMo & NemoClaw](https://img.shields.io/badge/NVIDIA_NeMo_/_Claw-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![SGLang Engine](https://img.shields.io/badge/SGLang-212121?style=for-the-badge&logo=speedtest&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

### ⚛️ Frontend UI & Simulation Stack
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![HTML5 & CSS3](https://img.shields.io/badge/HTML5_/_CSS3-E34F26?style=for-the-badge&logo=html5&logoColor=white)

### 📊 Data Science, Acceleration & Dev Environment
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![CuPy / PyCUDA](https://img.shields.io/badge/CuPy_/_PyCUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![VS Code](https://img.shields.io/badge/VS_Code-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white)

---

## 👥 Core Development Team (GigaChads)
* **Sharvin Mhatre** (ECS Dept., SIES Graduate School of Technology)
* **Archit Jaijith** (ECS Dept., SIES Graduate School of Technology)
* **Saumitra Chavan** (ECS Dept., SIES Graduate School of Technology)

---

## 🛠️ Full-Stack Modular Architecture

Our project is modularly split into a high-performance database processing engine and an interactive live telemetry dashboard to ensure a clean separation of concerns:

* **Backend Core & Agentic Memory Matrix (`SwarmDB`):** Handles $O(N \log N)$ cKD-Tree spatial partitioning layouts, parallel hash grid architectures, and the Track A multi-agent decentralized state synchronization orchestration engine (`agent_swarm_memory.py`).
* **Frontend Telemetry UI (`swarmdb-web`):** Built on Vite + React to deliver a high-impact, real-time particle simulation canvas, engine control panel, and live execution metrics graphing layout.

---

## ⚡ Actual SwarmDB Benchmarks

| Robots (N) | Naive CPU Baseline $O(N^2)$ | SwarmDB cKD-Tree Indexing $O(N \log N)$ | Memory Optimization Metric |
| :--- | :--- | :--- | :--- |
| **1,000** | 0.0228 s (22.90 MB) | 0.0015 s (0.06 MB) | 🚀 Ultra-low hardware overhead |
| **5,000** | 0.5147 s (572.24 MB) | 0.0036 s (0.06 MB) | 🚀 Ultra-low hardware overhead |
| **10,000** | 2.4970 s (2288.90 MB) | 0.0075 s (0.08 MB) | 🚀 Ultra-low hardware overhead |
| **50,000** | 💥 **CRASHED (OOM > 57 GB)** | ✨ **0.0497 s (0.52 MB)** | 📈 **28,600x Memory footprint reduction** |

---

## 🧠 Current Bottlenecks & Optimization Strategy

### 1. Traditional $O(N^2)$ Compute & Memory Limits
Calculating continuous pairwise distances requires a brute-force matrix expansion of $(A-B)^2 = A^2 + B^2 - 2A \cdot B$. As the entity array approaches 50,000 nodes, generating dense matrices triggers catastrophic Out-Of-Memory (OOM) failures on host CPU memory.

### 2. SGLang Inference Acceleration (Track A)
Traditional multi-agent text execution models face severe token-parsing overhead because lengthy context definitions and tool-states are continuously passed over standard APIs. SwarmDB integrates natively with **SGLang**, leveraging its **RadixAttention** engine to persistently cache static spatial index structures and system parameters inside the GPU memory matrix. The multi-agent workflow completely bypasses redundant calculation cycles.

### 3. Parallel CUDA Thread Layouts
We map spatial partitioning grids natively to GPU hardware. Individual coordinate components are assigned to independent **GPU thread blocks**, executing parallel distance comparisons across thousands of independent streaming multiprocessors concurrently. **GPU Shared Memory** arrays are implemented to eliminate data transfer lags across the PCIe bus.

---

## 🚀 Setup & Execution

### 1. System Requirements & Setup

Clone the engine repository and set up the local operational tracking layers:
```bash
pip install -r requirements.txt

2. Run the Multi-Agent State Synchronization Engine
To test the local-first decentralized memory agent workflow loop, run:

Bash
python agent_swarm_memory.py
3. Run the Native Performance Benchmarking Suite
To isolate and measure execution metrics across varying swarm densities:

Bash
python benchmark.py
Clone the engine repository and set up the local operational tracking layers:
```bash
pip install -r requirements.txt
