import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..ai_processor import AIProcessor
from ..web3_handler import Web3Handler
from ..analytics_manager import AnalyticsManager

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_ai_emotion_analysis():
    response = client.post(
        "/api/ai/emotion-analysis",
        json={"data": "I am very happy today!", "type": "text"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "emotion" in data
    assert "confidence" in data

def test_web3_connect():
    response = client.post(
        "/api/web3/connect",
        json={"wallet_address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "address" in data
    assert "network" in data

def test_analytics_log():
    response = client.post(
        "/api/analytics/log",
        json={
            "user_id": "test_user",
            "interaction_type": "test",
            "data": {"action": "click"}
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "interaction_id" in data

def test_analytics_stats():
    response = client.get("/api/analytics/stats")
    assert response.status_code == 200
    data = response.json()
    assert "active_users" in data
    assert "total_interactions" in data

@pytest.fixture
def ai_processor():
    return AIProcessor()

@pytest.fixture
def web3_handler():
    return Web3Handler()

@pytest.fixture
def analytics_manager():
    return AnalyticsManager()

def test_ai_processor(ai_processor):
    result = ai_processor.analyze_emotion("I am very happy today!")
    assert "emotion" in result
    assert "confidence" in result
    assert result["emotion"] in ["joy", "happy", "positive"]

def test_web3_handler(web3_handler):
    result = web3_handler.connect_wallet("0x742d35Cc6634C0532925a3b844Bc454e4438f44e")
    assert result["address"]
    assert result["network"] == "ethereum"

def test_analytics_manager(analytics_manager):
    result = analytics_manager.log_interaction({
        "user_id": "test_user",
        "type": "test",
        "data": {"action": "click"}
    })
    assert result["success"]
    assert "interaction_id" in result

def test_integration_flow():
    # Test complete user flow
    # 1. Connect wallet
    wallet_response = client.post(
        "/api/web3/connect",
        json={"wallet_address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"}
    )
    assert wallet_response.status_code == 200
    wallet_data = wallet_response.json()
    
    # 2. Analyze emotion
    emotion_response = client.post(
        "/api/ai/emotion-analysis",
        json={"data": "I am excited about Web3!", "type": "text"}
    )
    assert emotion_response.status_code == 200
    emotion_data = emotion_response.json()
    
    # 3. Log interaction
    log_response = client.post(
        "/api/analytics/log",
        json={
            "user_id": wallet_data["address"],
            "interaction_type": "emotion_analysis",
            "data": emotion_data
        }
    )
    assert log_response.status_code == 200
    
    # 4. Get statistics
    stats_response = client.get("/api/analytics/stats")
    assert stats_response.status_code == 200
    stats_data = stats_response.json()
    assert stats_data["total_interactions"] > 0
