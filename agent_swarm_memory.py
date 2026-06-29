import os
import time
import chromadb  # type: ignore
from chromadb.config import Settings  # type: ignore
from typing import Dict, Any, List, Optional

class SwarmDBNode:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.cluster_peers: List['SwarmDBNode'] = []
        
        # Initialize a true persistent vector database directory layout
        self.db_path = os.path.join(os.getcwd(), "swarm_db_vault", self.node_id)
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        
        # Isolate a structural collection instance for this specific node
        self.collection = self.chroma_client.get_or_create_collection(
            name=f"swarm_memory_{node_id.lower()}"
        )

    def connect_peer(self, peer_node: 'SwarmDBNode'):
        if peer_node not in self.cluster_peers:
            self.cluster_peers.append(peer_node)
            peer_node.cluster_peers.append(self)

    def write_state(self, key: str, value: str, broadcast: bool = True):
        """
        Saves unstructured state data along with automatic unique document 
        vector mappings, then handles decentralized peer replication blocks.
        """
        timestamp_str = str(time.time())
        
        # Write to persistent local vector space indices
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
        """ Decentralized sync listener mapping network metadata """
        self.collection.upsert(
            ids=[key],
            documents=[value],
            metadatas=[{"origin_node": origin_id, "timestamp": str(time.time())}]
        )
        print(f"⚡ [SwarmDB-{self.node_id}] Synchronized Vector state from Peer '{origin_id}' for key: '{key}'")

    def read_state(self, key: str) -> Optional[str]:
        """ Direct lookup via document primary tracking key identifiers """
        try:
            result = self.collection.get(ids=[key])
            if result and result['documents']:
                return result['documents'][0]
            return None
        except Exception:
            return None

    def query_semantic_memory(self, prompt_query: str, n_results: int = 1) -> List[str]:
        """
        Executes a true mathematical vector similarity search. This returns past 
        context vectors whose embeddings align nearest to the active agent query.
        """
        try:
            results = self.collection.query(
                query_texts=[prompt_query],
                n_results=n_results
            )
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
            print(f"❌ Semantic query bottleneck hit: {str(e)}")
            return []