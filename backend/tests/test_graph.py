import pytest
import sys
import os
import neo4j
from fastapi.testclient import TestClient
from main import app  # Import app, but handle driver in fixtures

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add parent directory to path

# Set up test client
# client = TestClient(app)  # Removed this line as it's now handled in fixtures

# Mock or set API key for tests (use environment variable in real scenarios)
# os.environ['API_KEY'] = 'test_key'  # Temporary for testing; should be handled securely

@pytest.fixture(scope='module')
def test_driver():
    # Use a test Neo4j URI or mock; for now, assume a test database is set up
    test_uri = 'bolt://localhost:7687'  # Change to a test instance if available
    test_driver = neo4j.GraphDatabase.driver(test_uri, auth=('test_user', 'test_password'))  # Use test credentials
    yield test_driver
    test_driver.close()

@pytest.fixture(scope='module')
def client(test_driver):
    # Override the app's driver with the test driver if possible, or ensure isolation
    # For simplicity, we're using the app as is, but in a real scenario, mock dependencies
    return TestClient(app)

def test_update_node(client):
    response = client.post("/api/graph/update-node", json={"node_id": "test_node_id", "label": "Test Label", "valence": 0.5}, headers={"X-API-Key": "test_key"})
    assert response.status_code == 200
    assert response.json().get("status") == "node updated"

def test_update_feedback(client):
    response = client.post("/api/graph/update-feedback", json={"node_id": "test_node_id", "feedback": "positive", "weight_change": 0.1}, headers={"X-API-Key": "test_key"})
    assert response.status_code == 200
    assert response.json().get("status") == "graph updated based on feedback"

def test_nlp_process_and_update(client):
    response = client.post("/api/nlp/process-and-update", json="I am happy about this.", headers={"X-API-Key": "test_key"})
    assert response.status_code == 200
    assert "graph updated" in response.json().get("status")
    assert len(response.json().get("entities_found", [])) > 0  # Expect entities like 'happy'
