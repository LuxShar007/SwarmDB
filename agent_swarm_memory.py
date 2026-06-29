import os
import time
import json
import chromadb
from typing import Dict, Any, List, Optional

class SwarmDBNode:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.cluster_peers: List['SwarmDBNode'] = []
        
        # Build a persistent directory context path on disk
        self.db_path = os.path.join(os.getcwd(), "swarm_db_vault", self.node_id)
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=f"swarm_memory_{node_id.lower()}"
        )

    def connect_peer(self, peer_node: 'SwarmDBNode'):
        if peer_node not in self.cluster_peers:
            self.cluster_peers.append(peer_node)
            peer_node.cluster_peers.append(self)

    def write_state(self, key: str, value: str, broadcast: bool = True):
        timestamp_str = str(time.time())
        
        # Commit vector metadata frames to persistent local storage segments
        self.collection.upsert(
            ids=[key],
            documents=[value],
            metadatas=[{"origin_node": self.node_id, "timestamp": timestamp_str}]
        )
        print(f"🔒 [SwarmDB-VectorVault-{self.node_id}] State Written: '{key}'")
        
        if broadcast:
            for peer in self.cluster_peers:
                peer.receive_sync(key, value, self.node_id)

    def receive_sync(self, key: str, value: str, origin_id: str):
        self.collection.upsert(
            ids=[key],
            documents=[value],
            metadatas=[{"origin_node": origin_id, "timestamp": str(time.time())}]
        )
        print(f"⚡ [SwarmDB-{self.node_id}] Sync from Peer '{origin_id}' locked for: '{key}'")

    def read_state(self, key: str) -> Optional[str]:
        try:
            result = self.collection.get(ids=[key])
            if result and result['documents']:
                return result['documents'][0]
            return None
        except Exception:
            return None

    def query_semantic_memory(self, prompt_query: str, n_results: int = 1) -> List[str]:
        """ Executes similarity matching vectors against previous historical text states """
        try:
            results = self.collection.query(
                query_texts=[prompt_query],
                n_results=n_results
            )
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
            return []