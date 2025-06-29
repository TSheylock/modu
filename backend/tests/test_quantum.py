import pytest
from fastapi.testclient import TestClient
from ..quantum.memory_core import QuantumMemoryCore
from ..routers.quantum_router import router
from ..services.web3_service import Web3Service
from eth_account import Account
from eth_account.messages import encode_defunct
import json
import networkx as nx
from datetime import datetime

# Test client setup
client = TestClient(router)

@pytest.fixture
def memory_core():
    return QuantumMemoryCore()

@pytest.fixture
def web3_service():
    config = {
        'web3': {
            'provider_url': 'http://localhost:8545',
            'chain_id': 1337
        }
    }
    return Web3Service(config)

@pytest.fixture
def test_accounts():
    return [Account.create() for _ in range(3)]

class TestQuantumMemoryCore:
    def test_initialization(self, memory_core):
        """Test memory core initialization"""
        assert isinstance(memory_core.graph, nx.Graph)
        assert memory_core.encryption_key is not None
        assert memory_core.cipher is not None
        
        # Check root nodes
        root_categories = ["emotion", "perception", "interaction", "knowledge"]
        for category in root_categories:
            assert f"root_{category}" in memory_core.graph.nodes

    def test_add_node(self, memory_core):
        """Test adding nodes to memory core"""
        test_node = "test_node"
        test_category = "test"
        test_metadata = {"key": "value"}
        
        memory_core.add_node(test_node, test_category, test_metadata)
        
        assert test_node in memory_core.graph.nodes
        assert memory_core.graph.nodes[test_node]["category"] == test_category
        assert memory_core.graph.nodes[test_node]["metadata"] == test_metadata

    def test_add_connection(self, memory_core):
        """Test adding connections between nodes"""
        node1 = "node1"
        node2 = "node2"
        weight = 0.75
        
        memory_core.add_node(node1)
        memory_core.add_node(node2)
        memory_core.add_connection(node1, node2, weight)
        
        assert memory_core.graph.has_edge(node1, node2)
        assert memory_core.graph[node1][node2]["weight"] == weight

    def test_quantum_perception(self, memory_core):
        """Test quantum perception processing"""
        test_input = "Test perception input"
        result = memory_core.quantum_perception(test_input)
        
        assert "encrypted_input" in result
        assert "key_hash" in result
        assert "quantum_basis" in result
        assert "timestamp" in result
        assert len(result["quantum_basis"]) == 32

    @pytest.mark.asyncio
    async def test_store_perception(self, memory_core, test_accounts):
        """Test storing perception data"""
        user = test_accounts[0]
        test_input = "Test perception"
        
        # Generate perception data
        perception_data = memory_core.quantum_perception(test_input)
        
        # Sign the data
        message = json.dumps(perception_data)
        msg_hash = encode_defunct(text=message)
        signature = user.sign_message(msg_hash).signature.hex()
        
        # Store perception
        result = await memory_core.store_perception(
            user.address,
            perception_data,
            signature
        )
        
        assert result["success"] is True
        assert "node_name" in result
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_retrieve_perception(self, memory_core, test_accounts):
        """Test retrieving perception data"""
        user = test_accounts[0]
        test_input = "Test perception"
        
        # Store perception first
        perception_data = memory_core.quantum_perception(test_input)
        message = json.dumps(perception_data)
        msg_hash = encode_defunct(text=message)
        signature = user.sign_message(msg_hash).signature.hex()
        
        store_result = await memory_core.store_perception(
            user.address,
            perception_data,
            signature
        )
        
        # Retrieve perception
        retrieve_message = f"retrieve_{store_result['node_name']}"
        retrieve_hash = encode_defunct(text=retrieve_message)
        retrieve_signature = user.sign_message(retrieve_hash).signature.hex()
        
        result = await memory_core.retrieve_perception(
            store_result["node_name"],
            user.address,
            retrieve_signature
        )
        
        assert result["success"] is True
        assert "perception" in result
        assert "context" in result
        assert result["perception"]["encrypted_input"] == perception_data["encrypted_input"]

class TestQuantumRouter:
    @pytest.mark.asyncio
    async def test_process_perception_endpoint(self, test_accounts, web3_service):
        """Test perception processing endpoint"""
        user = test_accounts[0]
        
        # Register user on blockchain first
        await web3_service.register_user(user.address)
        
        # Create and sign perception request
        request_data = {
            "user_input": "Test perception input",
            "user_address": user.address,
            "signature": "0x" + "0" * 130  # Mock signature
        }
        
        response = client.post("/quantum/perception", json=request_data)
        assert response.status_code == 200
        assert "success" in response.json()
        assert "perception_result" in response.json()

    @pytest.mark.asyncio
    async def test_retrieve_perception_endpoint(self, test_accounts, web3_service):
        """Test perception retrieval endpoint"""
        user = test_accounts[0]
        
        # Register user on blockchain first
        await web3_service.register_user(user.address)
        
        # Store perception first
        store_response = client.post("/quantum/perception", json={
            "user_input": "Test perception input",
            "user_address": user.address,
            "signature": "0x" + "0" * 130  # Mock signature
        })
        
        node_name = store_response.json()["perception_result"]["node_name"]
        
        # Retrieve perception
        retrieve_response = client.post("/quantum/retrieve", json={
            "node_name": node_name,
            "user_address": user.address,
            "signature": "0x" + "0" * 130  # Mock signature
        })
        
        assert retrieve_response.status_code == 200
        assert "success" in retrieve_response.json()
        assert "perception" in retrieve_response.json()

    def test_get_memory_state_endpoint(self):
        """Test getting memory state endpoint"""
        response = client.get("/quantum/memory")
        assert response.status_code == 200
        assert "nodes" in response.json()
        assert "edges" in response.json()
        assert "stats" in response.json()

    def test_health_check_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/quantum/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert "nodes_count" in response.json()
        assert "edges_count" in response.json()
        assert "timestamp" in response.json()

if __name__ == "__main__":
    pytest.main(["-v"])
