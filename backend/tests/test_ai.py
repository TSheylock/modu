import pytest
import asyncio
from ..ai.ai_manager import AIManager
from ..ai.emotion_detector import EmotionDetector
from ..ai.nlp_processor import NLPProcessor
from ..ai.learning_system import LearningSystem
import base64
import numpy as np
from datetime import datetime

@pytest.fixture
async def ai_manager():
    manager = AIManager()
    return manager

@pytest.fixture
async def emotion_detector():
    detector = EmotionDetector()
    return detector

@pytest.fixture
async def nlp_processor():
    processor = NLPProcessor()
    return processor

@pytest.fixture
async def learning_system():
    system = LearningSystem()
    return system

# AI Manager Tests
@pytest.mark.asyncio
async def test_ai_manager_initialization(ai_manager):
    assert ai_manager.emotion_detector is not None
    assert ai_manager.nlp_processor is not None
    assert ai_manager.learning_system is not None

@pytest.mark.asyncio
async def test_process_text(ai_manager):
    text_input = {
        'type': 'text',
        'data': 'I am feeling happy today!'
    }
    result = await ai_manager.process_input(text_input)
    
    assert result['success'] is True
    assert 'nlp_analysis' in result
    assert 'response' in result

@pytest.mark.asyncio
async def test_system_status(ai_manager):
    status = await ai_manager.get_system_status()
    
    assert status['success'] is True
    assert 'components' in status
    assert 'status' in status
    assert status['status'] == 'operational'

# Emotion Detector Tests
@pytest.mark.asyncio
async def test_emotion_detector_initialization(emotion_detector):
    model_info = emotion_detector.get_model_info()
    assert 'emotions_supported' in model_info
    assert model_info['model_loaded'] is True

@pytest.mark.asyncio
async def test_process_image(emotion_detector):
    # Create a simple test image
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Convert to base64
    _, buffer = cv2.imencode('.jpg', test_image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    result = await emotion_detector.process_image(image_base64)
    assert 'success' in result

# NLP Processor Tests
@pytest.mark.asyncio
async def test_nlp_processor_initialization(nlp_processor):
    model_info = nlp_processor.get_model_info()
    assert 'sentiment_analyzer' in model_info
    assert 'intent_classifier' in model_info
    assert 'response_generator' in model_info

@pytest.mark.asyncio
async def test_sentiment_analysis(nlp_processor):
    text = "I am very happy today!"
    result = await nlp_processor.analyze_sentiment(text)
    
    assert 'label' in result
    assert 'score' in result
    assert isinstance(result['score'], float)

@pytest.mark.asyncio
async def test_intent_classification(nlp_processor):
    text = "What is the weather like today?"
    result = await nlp_processor.classify_intent(text)
    
    assert 'intent' in result
    assert 'confidence' in result
    assert isinstance(result['confidence'], float)

# Learning System Tests
@pytest.mark.asyncio
async def test_learning_system_initialization(learning_system):
    assert learning_system.interaction_history == []
    assert isinstance(learning_system.model_performance, dict)

@pytest.mark.asyncio
async def test_log_interaction(learning_system):
    interaction_data = {
        'type': 'text',
        'input': 'Test input',
        'output': 'Test output',
        'feedback': 'positive'
    }
    
    result = await learning_system.log_interaction(interaction_data)
    assert result['success'] is True
    assert 'interaction_id' in result

@pytest.mark.asyncio
async def test_get_learning_stats(learning_system):
    stats = await learning_system.get_learning_stats()
    
    assert 'total_interactions' in stats
    assert 'training_data_size' in stats
    assert 'model_performance' in stats

# Integration Tests
@pytest.mark.asyncio
async def test_complete_ai_workflow(ai_manager):
    # Test text processing workflow
    text_input = {
        'type': 'text',
        'data': 'I am excited about learning AI!'
    }
    
    # Process input
    result = await ai_manager.process_input(text_input)
    assert result['success'] is True
    
    # Process feedback
    feedback = {
        'input_id': 1,
        'rating': 5,
        'comment': 'Great response!',
        'timestamp': datetime.utcnow().isoformat()
    }
    
    feedback_result = await ai_manager.process_feedback(feedback)
    assert feedback_result['success'] is True
    
    # Check system status after interaction
    status = await ai_manager.get_system_status()
    assert status['success'] is True
    assert status['status'] == 'operational'

@pytest.mark.asyncio
async def test_error_handling(ai_manager):
    # Test with invalid input
    invalid_input = {
        'type': 'invalid',
        'data': 'test'
    }
    
    result = await ai_manager.process_input(invalid_input)
    assert result['success'] is False
    assert 'error' in result

if __name__ == '__main__':
    pytest.main(['-v'])
