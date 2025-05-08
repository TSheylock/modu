"""
API для работы с семантической сетью SASOK

Предоставляет REST endpoints для доступа к базам знаний, 
эмоциональным датасетам и семантической сети SASOK.
"""
import os
import logging
import json
import asyncio
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from sasok_core.knowledge.semantic_network import SemanticNetwork
from sasok_core.utils.nats_client import NatsClient

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SASOK.KnowledgeAPI")

# Инициализация роутера FastAPI
router = APIRouter(prefix="/api/knowledge", tags=["Knowledge"])

# Глобальный экземпляр семантической сети
semantic_network = None

# Модели данных для API
class ImportRequest(BaseModel):
    """Запрос на импорт знаний"""
    source: str = Field(..., description="Источник знаний (conceptnet, wordnet, wikidata, emotion_datasets, all)")
    params: Dict[str, Any] = Field(default_factory=dict, description="Параметры импорта")

class SearchRequest(BaseModel):
    """Запрос на поиск в семантической сети"""
    query: str = Field(..., description="Поисковый запрос")
    sources: List[str] = Field(default=["all"], description="Источники для поиска (conceptnet, wordnet, wikidata, all)")
    params: Dict[str, Any] = Field(default_factory=dict, description="Параметры поиска")

class EmotionRequest(BaseModel):
    """Запрос на эмоциональный анализ"""
    concept: str = Field(..., description="Концепт для анализа")
    emotion_type: str = Field(default="basic", description="Тип эмоций (basic, complex, all)")

class TextEmotionRequest(BaseModel):
    """Запрос на эмоциональный анализ текста"""
    text: str = Field(..., description="Текст для анализа")
    detailed: bool = Field(default=False, description="Детализированный анализ")

# Инициализация семантической сети
async def initialize_semantic_network():
    """
    Инициализация глобального экземпляра семантической сети
    """
    global semantic_network
    
    # Получение настроек из переменных окружения
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    
    # Создание и инициализация семантической сети
    semantic_network = SemanticNetwork(neo4j_uri, neo4j_user, neo4j_password)
    await semantic_network.initialize()
    
    logger.info("Семантическая сеть инициализирована")

# Получение экземпляра семантической сети
async def get_semantic_network():
    """
    Получение экземпляра семантической сети
    
    Returns:
        Экземпляр SemanticNetwork
    """
    global semantic_network
    
    if semantic_network is None:
        await initialize_semantic_network()
    
    return semantic_network

# Эндпоинты API
@router.get("/status")
async def get_status(network: SemanticNetwork = Depends(get_semantic_network)):
    """
    Получение статуса семантической сети
    """
    return {
        "status": "active" if network and network.graph else "inactive",
        "initialized": network is not None,
        "timestamp": time.time()
    }

@router.post("/import")
async def import_knowledge(
    request: ImportRequest,
    background_tasks: BackgroundTasks,
    network: SemanticNetwork = Depends(get_semantic_network)
):
    """
    Запуск импорта знаний
    """
    # Проверка валидности источника
    valid_sources = ["conceptnet", "wordnet", "wikidata", "emotion_datasets", "all"]
    if request.source not in valid_sources:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Неверный источник знаний. Допустимые значения: {', '.join(valid_sources)}"
        )
    
    # Создание задачи импорта знаний
    task_id = f"import_{request.source}_{int(time.time())}"
    
    # Эмулируем сообщение NATS
    class MockMessage:
        def __init__(self, data):
            self.data = data.encode()
    
    msg = MockMessage(json.dumps({
        "source": request.source,
        "params": request.params
    }))
    
    # Запуск задачи в фоновом режиме
    background_tasks.add_task(network.handle_import_request, msg)
    
    return {
        "task_id": task_id,
        "source": request.source,
        "status": "started",
        "message": f"Задача импорта знаний из {request.source} запущена в фоновом режиме"
    }

@router.post("/search")
async def search_knowledge(
    request: SearchRequest,
    network: SemanticNetwork = Depends(get_semantic_network)
):
    """
    Поиск в семантической сети
    """
    # Проверка валидности источников
    valid_sources = ["conceptnet", "wordnet", "wikidata", "all"]
    for source in request.sources:
        if source not in valid_sources:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Неверный источник знаний. Допустимые значения: {', '.join(valid_sources)}"
            )
    
    # Эмулируем сообщение NATS
    class MockMessage:
        def __init__(self, data):
            self.data = data.encode()
            self.reply = None
    
    msg = MockMessage(json.dumps({
        "query": request.query,
        "sources": request.sources,
        "params": request.params
    }))
    
    # Перехватываем ответ
    original_publish = network.nats_client.publish
    
    results = None
    
    async def mock_publish(subject, data):
        nonlocal results
        if subject == "sasok.knowledge.search.response":
            results = json.loads(data.decode())
        else:
            await original_publish(subject, data)
    
    # Подменяем метод publish для перехвата ответа
    network.nats_client.publish = mock_publish
    
    # Выполняем поиск
    await network.handle_search_request(msg)
    
    # Восстанавливаем оригинальный метод
    network.nats_client.publish = original_publish
    
    # Возвращаем результаты
    if results and "error" in results:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=results["error"]
        )
    
    return results or {
        "query": request.query,
        "sources": request.sources,
        "results": {},
        "timestamp": time.time()
    }

@router.post("/emotion/concept")
async def analyze_concept_emotion(
    request: EmotionRequest,
    network: SemanticNetwork = Depends(get_semantic_network)
):
    """
    Эмоциональный анализ концепта
    """
    # Проверка валидности типа эмоций
    valid_types = ["basic", "complex", "all"]
    if request.emotion_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Неверный тип эмоций. Допустимые значения: {', '.join(valid_types)}"
        )
    
    # Эмулируем сообщение NATS
    class MockMessage:
        def __init__(self, data):
            self.data = data.encode()
            self.reply = None
    
    msg = MockMessage(json.dumps({
        "concept": request.concept,
        "emotion_type": request.emotion_type
    }))
    
    # Перехватываем ответ
    original_publish = network.nats_client.publish
    
    results = None
    
    async def mock_publish(subject, data):
        nonlocal results
        if subject == "sasok.knowledge.emotion.response":
            results = json.loads(data.decode())
        else:
            await original_publish(subject, data)
    
    # Подменяем метод publish для перехвата ответа
    network.nats_client.publish = mock_publish
    
    # Выполняем анализ
    await network.handle_emotion_request(msg)
    
    # Восстанавливаем оригинальный метод
    network.nats_client.publish = original_publish
    
    # Возвращаем результаты
    if results and "error" in results:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=results["error"]
        )
    
    return results or {
        "concept": request.concept,
        "emotion_type": request.emotion_type,
        "emotion_profile": {},
        "timestamp": time.time()
    }

@router.post("/emotion/text")
async def analyze_text_emotion(
    request: TextEmotionRequest,
    network: SemanticNetwork = Depends(get_semantic_network)
):
    """
    Эмоциональный анализ текста
    """
    # Выполняем анализ
    emotion_profile = await network.build_emotional_profile(request.text)
    
    # Формируем результат
    result = {
        "text": request.text[:100] + "..." if len(request.text) > 100 else request.text,
        "emotion_profile": emotion_profile,
        "timestamp": time.time()
    }
    
    # Если требуется детализированный анализ
    if request.detailed:
        # Анализ текста и выделение ключевых концептов (упрощенно)
        import re
        from collections import Counter
        
        # Токенизация и очистка текста
        words = re.findall(r'\b\w+\b', request.text.lower())
        
        # Фильтрация стоп-слов (упрощенно)
        stop_words = {"и", "в", "на", "с", "по", "для", "не", "от", "к", "у", "из", "а", "о", "что", "это"}
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Подсчет частоты слов
        word_counts = Counter(filtered_words)
        
        # Получение наиболее частых слов
        top_words = [word for word, count in word_counts.most_common(5)]
        
        # Анализ эмоций для каждого слова
        word_emotions = {}
        
        for word in top_words:
            word_profile = await network.analyze_concept_emotions(word)
            word_emotions[word] = word_profile
        
        result["detailed_analysis"] = {
            "key_concepts": top_words,
            "concept_emotions": word_emotions
        }
    
    return result

@router.get("/config")
async def get_config():
    """
    Получение конфигурации системы знаний
    """
    return {
        "neo4j_uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "sources": ["conceptnet", "wordnet", "wikidata", "emotion_datasets"],
        "emotion_types": ["basic", "complex", "all"],
        "basic_emotions": ["радость", "грусть", "злость", "страх", "удивление", "отвращение"],
        "status": "configured"
    }

# Обработчики событий FastAPI
@router.on_event("startup")
async def startup_event():
    """
    Обработчик события запуска API
    """
    # Инициализация семантической сети
    await initialize_semantic_network()

@router.on_event("shutdown")
async def shutdown_event():
    """
    Обработчик события остановки API
    """
    global semantic_network
    
    # Закрытие соединений
    if semantic_network:
        semantic_network.close()
        semantic_network = None
        
        logger.info("Семантическая сеть остановлена")
