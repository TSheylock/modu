import pytest
import torch
import numpy as np
from ..ml.data_pipeline import MultiModalPipeline
from ..ml.training_pipeline import TrainingPipeline
from ..ml.dataset_manager import DatasetManager
import os
import json
from datetime import datetime

@pytest.fixture
def config():
    return {
        'batch_size': 32,
        'num_epochs': 5,
        'input_dim': 768,  # BERT base hidden size
        'num_classes': 27,  # GoEmotions classes
        'ipfs_api_url': '/ip4/127.0.0.1/tcp/5001'
    }

@pytest.fixture
def data_pipeline():
    return MultiModalPipeline()

@pytest.fixture
def training_pipeline(config):
    return TrainingPipeline(config)

@pytest.fixture
def dataset_manager(config):
    return DatasetManager(config)

class TestDataPipeline:
    @pytest.mark.asyncio
    async def test_text_processing(self, data_pipeline):
        text = "I am feeling happy today!"
        result = await data_pipeline.process_text(text)
        
        assert result['success'] is True
        assert 'embeddings' in result
        assert isinstance(result['embeddings'], np.ndarray)
        assert 'text_length' in result
        assert 'timestamp' in result

    @pytest.mark.asyncio
    async def test_audio_processing(self, data_pipeline, tmp_path):
        # Create dummy audio file
        audio_path = tmp_path / "test_audio.wav"
        sample_rate = 16000
        duration = 2  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        import scipy.io.wavfile as wav
        wav.write(audio_path, sample_rate, audio_data.astype(np.float32))

        result = await data_pipeline.process_audio(str(audio_path))
        
        assert result['success'] is True
        assert 'features' in result
        assert isinstance(result['features'], np.ndarray)
        assert 'duration' in result
        assert 'timestamp' in result

    @pytest.mark.asyncio
    async def test_multimodal_processing(self, data_pipeline, tmp_path):
        # Prepare test data
        text = "I am feeling happy today!"
        audio_path = tmp_path / "test_audio.wav"
        sample_rate = 16000
        duration = 2
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t)
        import scipy.io.wavfile as wav
        wav.write(audio_path, sample_rate, audio_data.astype(np.float32))

        data = {
            'text': text,
            'audio': str(audio_path)
        }

        result = await data_pipeline.process_multimodal(data)
        
        assert result['success'] is True
        assert 'individual_results' in result
        assert 'combined_features' in result
        assert isinstance(result['combined_features'], np.ndarray)

class TestTrainingPipeline:
    def test_model_initialization(self, training_pipeline):
        # Create dummy data
        features = np.random.randn(100, training_pipeline.config['input_dim'])
        labels = np.random.randint(0, training_pipeline.config['num_classes'], 100)

        train_loader, val_loader = training_pipeline.prepare_data(features, labels)
        
        assert train_loader is not None
        assert val_loader is not None

    @pytest.mark.asyncio
    async def test_training(self, training_pipeline):
        # Create dummy data
        features = np.random.randn(100, training_pipeline.config['input_dim'])
        labels = np.random.randint(0, training_pipeline.config['num_classes'], 100)

        result = training_pipeline.train(features, labels, n_trials=2)
        
        assert result['success'] is True
        assert 'best_params' in result
        assert 'training_history' in result
        assert 'final_metrics' in result

    def test_model_save_load(self, training_pipeline, tmp_path):
        # Create and train model
        features = np.random.randn(100, training_pipeline.config['input_dim'])
        labels = np.random.randint(0, training_pipeline.config['num_classes'], 100)
        training_pipeline.train(features, labels, n_trials=2)

        # Save model
        save_path = tmp_path / "test_model.pt"
        save_result = training_pipeline.save_model(str(save_path))
        assert save_result['success'] is True

        # Load model
        load_result = training_pipeline.load_model(str(save_path))
        assert load_result['success'] is True
        assert 'config' in load_result
        assert 'best_params' in load_result
        assert 'training_history' in load_result

class TestDatasetManager:
    @pytest.mark.asyncio
    async def test_dataset_loading(self, dataset_manager):
        result = await dataset_manager.load_dataset("GoEmotions")
        
        assert result['success'] is True
        assert 'metadata' in result
        assert 'data' in result
        assert all(split in result['data'] for split in ['train', 'dev', 'test'])

    @pytest.mark.asyncio
    async def test_ipfs_save_load(self, dataset_manager):
        # Create dummy data
        data = {
            "features": np.random.randn(100, 768).tolist(),
            "labels": np.random.randint(0, 27, 100).tolist()
        }

        # Save to IPFS
        save_result = await dataset_manager.save_to_ipfs(data, "test_data")
        assert save_result['success'] is True
        assert 'ipfs_hash' in save_result

        # Load from IPFS
        load_result = await dataset_manager.load_from_ipfs(save_result['ipfs_hash'])
        assert load_result['success'] is True
        assert 'data' in load_result
        assert 'metadata' in load_result

    @pytest.mark.asyncio
    async def test_training_data_preparation(self, dataset_manager):
        # Load dataset first
        await dataset_manager.load_dataset("GoEmotions")

        # Prepare training data
        result = await dataset_manager.prepare_training_data("GoEmotions", "train")
        
        assert result['success'] is True
        assert 'features' in result
        assert 'labels' in result
        assert 'ids' in result

class TestIntegration:
    @pytest.mark.asyncio
    async def test_complete_pipeline(self, 
                                   data_pipeline, 
                                   training_pipeline, 
                                   dataset_manager):
        # 1. Load dataset
        dataset_result = await dataset_manager.load_dataset("GoEmotions")
        assert dataset_result['success'] is True

        # 2. Process text data
        text_data = dataset_result['data']['train']['texts'][:100]
        processed_results = []
        for text in text_data:
            result = await data_pipeline.process_text(text)
            assert result['success'] is True
            processed_results.append(result['embeddings'])

        features = np.vstack(processed_results)
        labels = dataset_result['data']['train']['emotions'][:100]

        # 3. Train model
        training_result = training_pipeline.train(features, labels, n_trials=2)
        assert training_result['success'] is True

        # 4. Save to IPFS
        save_result = await dataset_manager.save_to_ipfs(
            {
                "features": features.tolist(),
                "labels": labels.tolist(),
                "model_results": training_result
            },
            "integration_test"
        )
        assert save_result['success'] is True

if __name__ == "__main__":
    pytest.main(["-v"])
