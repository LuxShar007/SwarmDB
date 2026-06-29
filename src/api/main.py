import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from contextlib import asynccontextmanager
from agent_swarm_memory import SwarmDBNode
from src.kernel import kinetic_join_cuda
import numpy as np

# Global node dictionary collection context
db_nodes: Dict[str, SwarmDBNode] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 [SwarmDB] Unified API Engine Booting System Connections...")
    yield
    print("🛑 [SwarmDB] Shutting down database connections safely...")
    db_nodes.clear()

app = FastAPI(title="SwarmDB High-Throughput Cluster Hub API", lifespan=lifespan)

# Add CORS configuration layout boundaries to permit cross-origin front-end queries
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- REQUEST PAYLOAD COMPILATION GRID SCHEMAS ---
class MemoryWriteRequest(BaseModel):
    node_id: str
    key: str
    value: Any

class MemoryReadRequest(BaseModel):
    node_id: str
    key: str

class MemoryQueryRequest(BaseModel):
    node_id: str
    query_text: str
    n_results: int = 2

class PhysicsSimulationRequest(BaseModel):
    num_robots: int = 5000
    radius: float = 15.0

# --- ASYNCHRONOUS ENGINE ROUTE MAPPINGS ---
@app.post("/memory/write")
async def write_memory_state(req: MemoryWriteRequest):
    try:
        if req.node_id not in db_nodes:
            db_nodes[req.node_id] = SwarmDBNode(node_id=req.node_id)
        
        node = db_nodes[req.node_id]
        string_payload = json.dumps(req.value) if not isinstance(req.value, str) else req.value
        
        node.write_state(key=req.key, value=string_payload)
        return {"status": "success", "message": f"Payload committed under key '{req.key}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/read")
async def read_memory_state(req: MemoryReadRequest):
    if req.node_id not in db_nodes:
        raise HTTPException(status_code=404, detail=f"Node '{req.node_id}' not found.")
    try:
        node = db_nodes[req.node_id]
        value = node.read_state(key=req.key)
        try:
            return {"node_id": req.node_id, "key": req.key, "value": json.loads(value)}
        except (TypeError, json.JSONDecodeError):
            return {"node_id": req.node_id, "key": req.key, "value": value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/semantic-query")
async def semantic_query_memory(req: MemoryQueryRequest):
    if req.node_id not in db_nodes:
        raise HTTPException(status_code=404, detail=f"Node '{req.node_id}' not found.")
    try:
        node = db_nodes[req.node_id]
        matches = node.query_semantic_memory(prompt_query=req.query_text, n_results=req.n_results)
        return {"node_id": req.node_id, "query": req.query_text, "matches": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/swarm/physics-step")
async def execute_gpu_physics(req: PhysicsSimulationRequest):
    """ Direct pipeline execution path firing context straight down to your custom CUDA grid """
    try:
        # Generate spatial layout grids concurrently
        mock_positions = np.random.uniform(-500, 500, (req.num_robots, 3)).astype(np.float32)
        
        # Invoke your high-speed strided C++ hardware kernel calculation loop
        collision_pairs = kinetic_join_cuda(mock_positions, req.radius)
        
        return {
            "status": "success",
            "hardware_accelerator_utilized": "NVIDIA CUDA Core Architecture",
            "robots_processed": req.num_robots,
            "total_collisions_flagged": len(collision_pairs),
            "sample_pairs": collision_pairs[:5].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPU hardware backend track threw error: {str(e)}")