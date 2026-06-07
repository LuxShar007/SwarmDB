import json
import time
from typing import Dict, Any, List

# Simulating the core distributed node logic of your SwarmDB repository
class SwarmDBNode:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.local_memory_vault: Dict[str, Any] = {}
        self.cluster_peers: List['SwarmDBNode'] = []

    def connect_peer(self, peer_node: 'SwarmDBNode'):
        if peer_node not in self.cluster_peers:
            self.cluster_peers.append(peer_node)
            peer_node.cluster_peers.append(self)

    def write_state(self, key: str, value: Any, broadcast: bool = True):
        """ Writes state locally and handles decentralized synchronization across the swarm """
        self.local_memory_vault[key] = {
            "value": value,
            "timestamp": time.time(),
            "origin_node": self.node_id
        }
        print(f"🔒 [SwarmDB-{self.node_id}] State Written: '{key}' -> {value}")
        
        if broadcast:
            for peer in self.cluster_peers:
                peer.receive_sync(key, value, self.node_id)

    def receive_sync(self, key: str, value: Any, origin_id: str):
        """ Network synchronization layer matching SwarmDB architecture """
        self.local_memory_vault[key] = {
            "value": value,
            "timestamp": time.time(),
            "origin_node": origin_id
        }
        print(f"⚡ [SwarmDB-{self.node_id}] Synchronized state from Peer '{origin_id}' for key: '{key}'")

    def read_state(self, key: str) -> Any:
        state_entry = self.local_memory_vault.get(key)
        return state_entry["value"] if state_entry else None


# Track A Agent Framework Layer
class AutonomousAgent:
    def __init__(self, name: str, role: str, db_node: SwarmDBNode):
        self.name = name
        self.role = role
        self.memory_engine = db_node  # Under-the-hood SwarmDB integration

    def execute_task(self, task_description: str, input_context_key: str = None):
        print(f"\n🤖 Agent [{self.name}] ({self.role}) starting task: '{task_description}'")
        
        # Check if another agent has already cached critical parameters in SwarmDB
        if input_context_key:
            shared_context = self.memory_engine.read_state(input_context_key)
            if shared_context:
                print(f"💡 [{self.name}] Found pre-cached context in SwarmDB! Avoiding redundant LLM processing steps.")
                print(f"   Context Data: {shared_context}")
                return f"Processed successfully using shared memory: {shared_context}"
        
        # Simulating an LLM engine or Tool Execution output
        simulated_output = f"Optimized BOM Result for {task_description}"
        
        # Instantly update SwarmDB so all other agents receive the state update immediately
        output_key = f"result_{task_description.lower().replace(' ', '_')}"
        self.memory_engine.write_state(output_key, {"status": "Verified", "data": simulated_output})
        return simulated_output


# --- Execution Pipeline Simulation ---
if __name__ == "__main__":
    print("=====================================================================")
    print("⚡ DEMONSTRATING SWARMDB-AGENTIC DECENTRALIZED MEMORY INTERFACE ⚡")
    print("=====================================================================\n")

    # 1. Spin up independent database nodes acting as agent memory layers
    extractor_db = SwarmDBNode(node_id="ExtractorNode")
    scout_db = SwarmDBNode(node_id="ScoutNode")
    
    # Establish decentralized synchronization topology (connecting your peers)
    extractor_db.connect_peer(scout_db)

    # 2. Instantiate Track A Agents bound to individual SwarmDB memory nodes
    agent_extractor = AutonomousAgent(name="SchematicParser", role="Data Extraction", db_node=extractor_db)
    agent_scout = AutonomousAgent(name="InventoryScout", role="Logistics Tracking", db_node=scout_db)

    # 3. Step 1 of Workflow: Agent 1 extracts metrics and writes to its local DB node
    agent_extractor.execute_task(task_description="Parse Controller IC-74HC04")

    # 4. Step 2 of Workflow: Agent 2 automatically accesses the synchronized global state
    # Notice that agent_scout checks its own database instance ('scout_db') and gets the data
    # that was compiled by agent_extractor onto 'extractor_db' without redundant network loops!
    agent_scout.execute_task(
        task_description="Check live stock levels", 
        input_context_key="result_parse_controller_ic-74hc04"
    )