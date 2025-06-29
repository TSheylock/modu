import pytest
from fastapi.testclient import TestClient
from ..ml.dataset_loaders import DatasetLoaders
from ..ml.dataset_manager import DatasetManager
from ..routers.dataset_router import router
from ..services.web3_service import Web3Service
import os
import json
from pathlib import Path
import shutil
import asyncio
from datetime import datetime

# Test client setup
client = TestClient(router)

@pytest.fixture
def config():
    return {
        'ipfs_api_url': '/ip4/127.0.0.1/tcp/5001',
        'web3': {
            'provider_url': 'http://localhost:8545',
            'chain_id': 1337
        }
    }

@pytest.fixture
def dataset_loaders():
    # Use temporary directory for testing
    test_path = Path("test_data")
    loader = DatasetLoaders(base_path=test_path)
    yield loader
    # Cleanup
    if test_path.exists():
        shutil.rmtree(test_path)

@pytest.fixture
def dataset_manager(config):
    return DatasetManager(config)

@pytest.fixture
def web3_service(config):
    return Web3Service(config)

class TestDatasetLoaders:
    @pytest.mark.asyncio
    async def test_go_emotions_loading(self, dataset_loaders):
        """Test loading GoEmotions dataset"""
        result = await dataset_loaders.load_go_emotions()
        
        assert result['success'] is True
        assert 'data' in result
        assert all(split in result['data'] for split in ['train', 'dev', 'test'])
        
        # Check data structure
        train_data = result['data']['train']
        assert 'text' in train_data.columns
        assert 'emotions' in train_data.columns

    @pytest.mark.asyncio
    async def test_emobank_loading(self, dataset_loaders):
        """Test loading EmoBank dataset"""
        result = await dataset_loaders.load_emobank()
        
        assert result['success'] is True
        assert 'data' in result
        
        # Check data structure
        data = result['data']
        assert 'V' in data.columns  # Valence
        assert 'A' in data.columns  # Arousal
        assert 'D' in data.columns  # Dominance

    @pytest.mark.asyncio
    async def test_cornell_movie_dialogs_loading(self, dataset_loaders):
        """Test loading Cornell Movie Dialogs dataset"""
        result = await dataset_loaders.load_cornell_movie_dialogs()
        
        assert result['success'] is True
        assert 'data' in result

    def test_available_datasets(self, dataset_loaders):
        """Test getting list of available datasets"""
        datasets = dataset_loaders.get_available_datasets()
        
        assert isinstance(datasets, list)
        assert len(datasets) > 0
        assert "GoEmotions" in datasets
        assert "IEMOCAP" in datasets

class TestDatasetRouter:
    def test_get_available_datasets(self):
        """Test GET /datasets/available endpoint"""
        response = client.get("/datasets/available")
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert isinstance(data['datasets'], list)
        assert len(data['datasets']) > 0

    @pytest.mark.asyncio
    async def test_load_dataset(self):
        """Test POST /datasets/load endpoint"""
        response = client.post("/datasets/load", json={
            "name": "GoEmotions",
            "save_to_ipfs": True
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert "status_endpoint" in data

        # Check status
        status_response = client.get(data['status_endpoint'])
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data['dataset_name'] == "GoEmotions"
        assert 'status' in status_data
        assert 'progress' in status_data

    def test_get_dataset_info(self):
        """Test GET /datasets/info/{dataset_name} endpoint"""
        # First load a dataset
        client.post("/datasets/load", json={
            "name": "GoEmotions",
            "save_to_ipfs": False
        })
        
        # Get info
        response = client.get("/datasets/info/GoEmotions")
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert 'metadata' in data

    @pytest.mark.asyncio
    async def test_prepare_training_data(self):
        """Test POST /datasets/prepare endpoint"""
        # First load a dataset
        client.post("/datasets/load", json={
            "name": "GoEmotions",
            "save_to_ipfs": False
        })
        
        # Prepare training data
        response = client.post("/datasets/prepare", params={
            "dataset_name": "GoEmotions",
            "split": "train"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert 'features' in data
        assert 'labels' in data

class TestIntegration:
    @pytest.mark.asyncio
    async def test_complete_dataset_flow(self, 
                                       dataset_loaders, 
                                       dataset_manager,
                                       web3_service):
        """Test complete dataset workflow"""
        # 1. Load dataset
        load_result = await dataset_loaders.load_go_emotions()
        assert load_result['success'] is True

        # 2. Save to IPFS
        ipfs_result = await dataset_manager.save_to_ipfs(
            load_result['data'],
            "GoEmotions"
        )
        assert ipfs_result['success'] is True
        assert 'ipfs_hash' in ipfs_result

        # 3. Record on blockchain
        tx_result = await web3_service.record_interaction(
            "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",  # Test address
            "DATASET_LOAD",
            {
                "dataset": "GoEmotions",
                "ipfs_hash": ipfs_result['ipfs_hash'],
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        assert tx_result['success'] is True

        # 4. Prepare training data
        prep_result = await dataset_manager.prepare_training_data(
            "GoEmotions",
            "train"
        )
        assert prep_result['success'] is True
        assert 'features' in prep_result
        assert 'labels' in prep_result

    @pytest.mark.asyncio
    async def test_error_handling(self, dataset_loaders):
        """Test error handling in dataset operations"""
        # Test loading non-existent dataset
        result = await dataset_loaders.load_semeval()
        assert result['success'] is False
        assert 'error' in result

        # Test loading without required files
        result = await dataset_loaders.load_iemocap()
        assert result['success'] is False
        assert 'error' in result

if __name__ == "__main__":
    pytest.main(["-v"])
