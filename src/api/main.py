import os
import sys
import time
import tracemalloc
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Add workspace and src directory to python search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.main import RobotSwarm
from agent_swarm_memory import SwarmDBNode, AutonomousAgent

app = FastAPI(
    title="SwarmDB Spatial Indexing & Agent API Engine",
    description="GPU-accelerated Kinetic Join Spatial Database and Decentralized Agent Memory Layer",
    version="1.0.0"
)

# CORS configuration to allow connections from Vite/React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store swarm instance and node cache
swarm_instance: Optional[RobotSwarm] = None
db_nodes: Dict[str, SwarmDBNode] = {}

class SwarmInitRequest(BaseModel):
    num_robots: int
    min_bound: float = 0.0
    max_bound: float = 1000.0

class SwarmUpdateRequest(BaseModel):
    max_step: float = 1.0

class SwarmCollisionRequest(BaseModel):
    radius: float
    method: str = "optimized"  # "baseline", "optimized", "cuda"

class MemoryWriteRequest(BaseModel):
    node_id: str
    key: str
    value: Any

class MemoryReadRequest(BaseModel):
    node_id: str
    key: str

class ConnectPeersRequest(BaseModel):
    node_id_1: str
    node_id_2: str

@app.get("/")
def read_root():
    return {
        "status": "online",
        "engine": "SwarmDB Spatial Engine & Decentralized Memory Layer",
        "version": "1.0.0",
        "developers": ["Sharvin Mhatre", "Archit Jaijith", "Saumitrya Chavan"]
    }

@app.post("/swarm/initialize")
def initialize_swarm(req: SwarmInitRequest):
    global swarm_instance
    try:
        swarm_instance = RobotSwarm(
            num_robots=req.num_robots,
            bounds=(req.min_bound, req.max_bound)
        )
        return {
            "status": "success",
            "message": f"Initialized swarm with {req.num_robots} robots in bounds [{req.min_bound}, {req.max_bound}]",
            "num_robots": req.num_robots
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/swarm/update")
def update_swarm(req: SwarmUpdateRequest):
    global swarm_instance
    if swarm_instance is None:
        raise HTTPException(status_code=400, detail="Swarm not initialized. Call /swarm/initialize first.")
    try:
        swarm_instance.update_positions(max_step=req.max_step)
        return {
            "status": "success",
            "message": "Swarm positions updated"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/swarm/positions")
def get_swarm_positions():
    global swarm_instance
    if swarm_instance is None:
        raise HTTPException(status_code=400, detail="Swarm not initialized. Call /swarm/initialize first.")
    
    positions = swarm_instance.positions.tolist()
    return {
        "num_robots": len(positions),
        "positions": [{"id": i, "x": pos[0], "y": pos[1], "z": pos[2]} for i, pos in enumerate(positions)]
    }

@app.post("/swarm/collisions")
def find_collisions(req: SwarmCollisionRequest):
    global swarm_instance
    if swarm_instance is None:
        raise HTTPException(status_code=400, detail="Swarm not initialized. Call /swarm/initialize first.")
    
    try:
        tracemalloc.start()
        start_time = time.perf_counter()
        
        collisions = swarm_instance.find_collisions(radius=req.radius, method=req.method)
        
        end_time = time.perf_counter()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            "status": "success",
            "method": req.method,
            "radius": req.radius,
            "collision_count": len(collisions),
            "execution_time_ms": (end_time - start_time) * 1000,
            "peak_memory_mb": peak_mem / (1024 * 1024),
            "collisions": collisions.tolist()
        }
    except ImportError as ie:
        raise HTTPException(status_code=400, detail=f"CUDA execution not available: {str(ie)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/write")
def write_memory_state(req: MemoryWriteRequest):
    try:
        if req.node_id not in db_nodes:
            db_nodes[req.node_id] = SwarmDBNode(node_id=req.node_id)
        
        node = db_nodes[req.node_id]
        node.write_state(key=req.key, value=req.value)
        return {
            "status": "success",
            "message": f"Successfully wrote key '{req.key}' to node '{req.node_id}'",
            "entry": node.local_memory_vault.get(req.key)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/read")
def read_memory_state(req: MemoryReadRequest):
    if req.node_id not in db_nodes:
        raise HTTPException(status_code=404, detail=f"Node '{req.node_id}' does not exist.")
    try:
        node = db_nodes[req.node_id]
        value = node.read_state(key=req.key)
        return {
            "node_id": req.node_id,
            "key": req.key,
            "value": value
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/connect")
def connect_nodes(req: ConnectPeersRequest):
    try:
        if req.node_id_1 not in db_nodes:
            db_nodes[req.node_id_1] = SwarmDBNode(node_id=req.node_id_1)
        if req.node_id_2 not in db_nodes:
            db_nodes[req.node_id_2] = SwarmDBNode(node_id=req.node_id_2)
        
        node1 = db_nodes[req.node_id_1]
        node2 = db_nodes[req.node_id_2]
        node1.connect_peer(node2)
        
        return {
            "status": "success",
            "message": f"Peering connection established between '{req.node_id_1}' and '{req.node_id_2}'",
            "peers_node_1": [peer.node_id for peer in node1.cluster_peers],
            "peers_node_2": [peer.node_id for peer in node2.cluster_peers]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/swarm/benchmark")
def run_api_benchmark(radius: float = 5.0):
    """
    Executes the SwarmDB kinetic join benchmark suite and returns the metrics as JSON.
    """
    # Size lists matching benchmark.py
    num_robots_list = [1000, 2500, 5000]
    large_num_robots_list = [10000]
    
    results = []
    
    # Check for CUDA availability
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

    # 1. Benchmark CPU Baseline
    for n in num_robots_list:
        swarm = RobotSwarm(num_robots=n)
        tracemalloc.start()
        start = time.perf_counter()
        collisions = swarm.find_collisions(radius=radius, method="baseline")
        end = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        results.append({
            "robots": n,
            "method": "Baseline (O(N²))",
            "time_ms": (end - start) * 1000,
            "memory_mb": peak / (1024 * 1024),
            "collisions": len(collisions)
        })

    # 2. Benchmark CPU Optimized (cKDTree)
    for n in num_robots_list + large_num_robots_list:
        swarm = RobotSwarm(num_robots=n)
        tracemalloc.start()
        start = time.perf_counter()
        collisions = swarm.find_collisions(radius=radius, method="optimized")
        end = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        results.append({
            "robots": n,
            "method": "Optimized (cKDTree)",
            "time_ms": (end - start) * 1000,
            "memory_mb": peak / (1024 * 1024),
            "collisions": len(collisions)
        })

    # 3. Benchmark GPU CUDA (if available)
    cuda_error = None
    if cuda_available:
        try:
            for n in num_robots_list + large_num_robots_list:
                swarm = RobotSwarm(num_robots=n)
                tracemalloc.start()
                start = time.perf_counter()
                collisions = swarm.find_collisions(radius=radius, method="cuda")
                end = time.perf_counter()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                results.append({
                    "robots": n,
                    "method": "CUDA GPU (Shared)",
                    "time_ms": (end - start) * 1000,
                    "memory_mb": peak / (1024 * 1024),
                    "collisions": len(collisions)
                })
        except Exception as e:
            cuda_error = str(e)

    return {
        "status": "success",
        "cuda_supported": cuda_available,
        "cuda_error": cuda_error,
        "results": results
    }
