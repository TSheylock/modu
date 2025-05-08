import networkx as nx
from cryptography.fernet import Fernet
import random
import base64
import hashlib
import json
from eth_account import Account
from eth_account.messages import encode_defunct
from typing import Dict, List, Optional
import logging
from datetime import datetime

class QuantumMemoryCore:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.graph = nx.Graph()
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        self.initialize_core()

    def initialize_core(self):
        """Initialize the quantum memory core"""
        try:
            # Initialize base nodes for different types of memories
            base_categories = ["emotion", "perception", "interaction", "knowledge"]
            for category in base_categories:
                self.add_node(f"root_{category}", category, {
                    "created_at": datetime.utcnow().isoformat(),
                    "type": "root",
                    "description": f"Root node for {category} memories"
                })
            
            self.logger.info("Quantum memory core initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing quantum memory core: {str(e)}")
            raise

    def add_node(self, node_name: str, category: Optional[str] = None, metadata: Optional[Dict] = None) -> None:
        """Add a new node to the memory graph"""
        try:
            self.graph.add_node(node_name, 
                              category=category, 
                              metadata=metadata or {},
                              created_at=datetime.utcnow().isoformat())
            
            # Connect to category root if category is specified
            if category:
                root_node = f"root_{category}"
                if root_node in self.graph:
                    self.add_connection(node_name, root_node)
                    
        except Exception as e:
            self.logger.error(f"Error adding node: {str(e)}")
            raise

    def add_connection(self, node1: str, node2: str, weight: float = 1.0) -> None:
        """Add a weighted connection between nodes"""
        try:
            self.graph.add_edge(node1, node2, 
                              weight=weight,
                              created_at=datetime.utcnow().isoformat())
        except Exception as e:
            self.logger.error(f"Error adding connection: {str(e)}")
            raise

    def quantum_perception(self, user_input: str) -> Dict:
        """Process input through quantum-inspired perception"""
        try:
            # Implement BB84-like protocol for quantum-inspired encryption
            basis = [random.choice([0, 1]) for _ in range(256)]
            bits = [random.choice([0, 1]) for _ in range(256)]
            key = bytes([b if basis[i] == 0 else (1 - b) 
                        for i, b in enumerate(bits)][:32])
            key = base64.urlsafe_b64encode(key)

            # Encrypt the input
            cipher = Fernet(key)
            encrypted_input = cipher.encrypt(user_input.encode())
            
            return {
                "encrypted_input": encrypted_input.decode(),
                "key_hash": hashlib.sha256(key).hexdigest(),
                "quantum_basis": basis[:32],  # Store partial basis for verification
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error in quantum perception: {str(e)}")
            raise

    async def store_perception(self, 
                             user_address: str, 
                             perception_data: Dict, 
                             signature: str) -> Dict:
        """Store perception data with blockchain verification"""
        try:
            # Verify signature
            message = json.dumps(perception_data)
            msg_hash = encode_defunct(text=message)
            recovered_address = Account.recover_message(msg_hash, signature=signature)
            
            if recovered_address.lower() != user_address.lower():
                raise ValueError("Invalid signature")

            # Create perception node
            node_name = f"perception_{hashlib.sha256(message.encode()).hexdigest()}"
            
            self.add_node(node_name, "perception", {
                "user_address": user_address,
                "encrypted_data": perception_data["encrypted_input"],
                "key_hash": perception_data["key_hash"],
                "quantum_basis": perception_data["quantum_basis"],
                "timestamp": perception_data["timestamp"]
            })

            # Add connections to related nodes based on quantum similarity
            self._add_quantum_connections(node_name)

            return {
                "success": True,
                "node_name": node_name,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error storing perception: {str(e)}")
            raise

    async def retrieve_perception(self, 
                                node_name: str, 
                                user_address: str, 
                                signature: str) -> Dict:
        """Retrieve perception data with verification"""
        try:
            # Verify signature
            message = f"retrieve_{node_name}"
            msg_hash = encode_defunct(text=message)
            recovered_address = Account.recover_message(msg_hash, signature=signature)
            
            if recovered_address.lower() != user_address.lower():
                raise ValueError("Invalid signature")

            # Get node data
            node_data = self.graph.nodes.get(node_name)
            if not node_data or node_data["metadata"]["user_address"] != user_address:
                raise ValueError("Access denied")

            # Get connected nodes for context
            connected_nodes = list(self.graph.neighbors(node_name))
            
            return {
                "success": True,
                "perception": {
                    "encrypted_input": node_data["metadata"]["encrypted_data"],
                    "key_hash": node_data["metadata"]["key_hash"],
                    "quantum_basis": node_data["metadata"]["quantum_basis"],
                    "timestamp": node_data["metadata"]["timestamp"]
                },
                "context": {
                    "connected_nodes": connected_nodes,
                    "category": node_data["category"]
                }
            }

        except Exception as e:
            self.logger.error(f"Error retrieving perception: {str(e)}")
            raise

    def _add_quantum_connections(self, node_name: str) -> None:
        """Add quantum-inspired connections between related nodes"""
        try:
            node_data = self.graph.nodes[node_name]["metadata"]
            quantum_basis = node_data["quantum_basis"]
            
            # Find potential connections based on quantum similarity
            for other_node in self.graph.nodes:
                if other_node != node_name and "metadata" in self.graph.nodes[other_node]:
                    other_data = self.graph.nodes[other_node]["metadata"]
                    if "quantum_basis" in other_data:
                        # Calculate quantum similarity
                        similarity = self._calculate_quantum_similarity(
                            quantum_basis,
                            other_data["quantum_basis"]
                        )
                        
                        # Add connection if similarity is above threshold
                        if similarity > 0.7:  # Adjustable threshold
                            self.add_connection(node_name, other_node, weight=similarity)

        except Exception as e:
            self.logger.error(f"Error adding quantum connections: {str(e)}")
            raise

    def _calculate_quantum_similarity(self, basis1: List[int], basis2: List[int]) -> float:
        """Calculate similarity between quantum bases"""
        try:
            matching_positions = sum(1 for b1, b2 in zip(basis1, basis2) if b1 == b2)
            return matching_positions / len(basis1)
        except Exception as e:
            self.logger.error(f"Error calculating quantum similarity: {str(e)}")
            raise

    def export_to_json(self) -> Dict:
        """Export memory graph to JSON format"""
        try:
            nodes = [{
                "id": node,
                "label": node,
                "group": self.graph.nodes[node].get("category", "default"),
                "metadata": self.graph.nodes[node].get("metadata", {})
            } for node in self.graph.nodes]
            
            edges = [{
                "from": u,
                "to": v,
                "value": self.graph[u][v]["weight"],
                "created_at": self.graph[u][v].get("created_at")
            } for u, v in self.graph.edges]
            
            return {
                "nodes": nodes,
                "edges": edges,
                "stats": {
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                    "categories": list(set(node["group"] for node in nodes))
                }
            }
        except Exception as e:
            self.logger.error(f"Error exporting to JSON: {str(e)}")
            raise

    def get_node_connections(self, node_name: str) -> Dict:
        """Get all connections for a specific node"""
        try:
            if node_name not in self.graph:
                raise ValueError(f"Node {node_name} not found")
                
            connections = list(self.graph.neighbors(node_name))
            return {
                "node": node_name,
                "connections": [{
                    "node": conn,
                    "weight": self.graph[node_name][conn]["weight"],
                    "created_at": self.graph[node_name][conn].get("created_at")
                } for conn in connections]
            }
        except Exception as e:
            self.logger.error(f"Error getting node connections: {str(e)}")
            raise
