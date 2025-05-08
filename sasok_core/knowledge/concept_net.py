"""
Модуль интеграции ConceptNet в семантическую сеть SASOK

Загружает данные из ConceptNet и интегрирует их в Neo4j граф для 
последующего использования в эмоциональном анализе и контекстном понимании.
"""
import os
import logging
import json
import requests
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import time
import hashlib

import networkx as nx
from py2neo import Node, Relationship
from tqdm import tqdm

from sasok_core.knowledge.knowledge_base import KnowledgeBase
from sasok_core.utils.nats_client import NatsClient

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SASOK.ConceptNet")

# Путь к файлам данных
DATA_DIR = Path(os.getenv("DATA_DIR", "/home/sasok/Рабочий стол/blackboxai-1745739396945/data"))
CONCEPT_NET_DIR = DATA_DIR / "knowledge" / "conceptnet"
CONCEPT_NET_DIR.mkdir(parents=True, exist_ok=True)

# URL для API ConceptNet
CONCEPT_NET_API = "https://api.conceptnet.io"

class ConceptNetProcessor(KnowledgeBase):
    """
    Класс для обработки и интеграции данных из ConceptNet
    """
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, lang: str = "ru"):
        """
        Инициализация процессора ConceptNet
        
        Args:
            neo4j_uri: URI для подключения к Neo4j
            neo4j_user: Имя пользователя Neo4j
            neo4j_password: Пароль пользователя Neo4j
            lang: Основной язык для извлечения концептов
        """
        super().__init__(neo4j_uri, neo4j_user, neo4j_password)
        self.lang = lang
        self.nats_client = None
        self.concept_cache = {}
        
        # Создание индексов в Neo4j
        self.create_index("Concept", "id")
        self.create_index("Concept", "uri")
        self.create_index("Concept", "name")
    
    async def initialize(self):
        """
        Асинхронная инициализация, включая подключение к NATS
        """
        # Подключение к NATS
        nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
        self.nats_client = await NatsClient.get_instance(nats_url)
        logger.info("ConceptNetProcessor инициализирован и подключен к NATS")
    
    def get_concept(self, concept_uri: str) -> Dict[str, Any]:
        """
        Получение концепта из ConceptNet API
        
        Args:
            concept_uri: URI концепта
            
        Returns:
            Информация о концепте
        """
        # Проверка кэша
        if concept_uri in self.concept_cache:
            return self.concept_cache[concept_uri]
        
        # Если URI не начинается с /, добавляем его
        if not concept_uri.startswith('/'):
            concept_uri = f"/{concept_uri}"
        
        # Запрос к API
        try:
            url = f"{CONCEPT_NET_API}{concept_uri}"
            response = requests.get(url)
            if response.status_code == 200:
                concept_data = response.json()
                
                # Кэширование результата
                self.concept_cache[concept_uri] = concept_data
                
                return concept_data
            else:
                logger.warning(f"Ошибка получения концепта {concept_uri}: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Ошибка API ConceptNet: {e}")
            return {}
    
    def search_concept(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Поиск концептов в ConceptNet API
        
        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов
            
        Returns:
            Список найденных концептов
        """
        # Кэш-ключ для запроса
        cache_key = f"search:{query}:{limit}"
        if cache_key in self.concept_cache:
            return self.concept_cache[cache_key]
        
        # Запрос к API
        try:
            url = f"{CONCEPT_NET_API}/search?text={query}&limit={limit}"
            response = requests.get(url)
            if response.status_code == 200:
                search_data = response.json()
                
                # Кэширование результата
                self.concept_cache[cache_key] = search_data.get("edges", [])
                
                return search_data.get("edges", [])
            else:
                logger.warning(f"Ошибка поиска концептов '{query}': {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Ошибка API ConceptNet: {e}")
            return []
    
    def get_related_concepts(self, concept: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Получение связанных концептов
        
        Args:
            concept: Концепт для поиска связей
            limit: Максимальное количество результатов
            
        Returns:
            Список связанных концептов
        """
        # Форматирование концепта для API
        if not concept.startswith('/c/'):
            # Нормализация имени концепта
            concept_name = concept.lower().replace(' ', '_')
            concept_uri = f"/c/{self.lang}/{concept_name}"
        else:
            concept_uri = concept
        
        # Кэш-ключ для запроса
        cache_key = f"related:{concept_uri}:{limit}"
        if cache_key in self.concept_cache:
            return self.concept_cache[cache_key]
        
        # Запрос к API
        try:
            url = f"{CONCEPT_NET_API}/related{concept_uri}?limit={limit}"
            response = requests.get(url)
            if response.status_code == 200:
                related_data = response.json()
                
                # Кэширование результата
                self.concept_cache[cache_key] = related_data.get("related", [])
                
                return related_data.get("related", [])
            else:
                logger.warning(f"Ошибка получения связанных концептов '{concept}': {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Ошибка API ConceptNet: {e}")
            return []
    
    def import_concept_to_neo4j(self, concept_data: Dict[str, Any]) -> Optional[Node]:
        """
        Импорт концепта в Neo4j
        
        Args:
            concept_data: Данные концепта
            
        Returns:
            Узел Neo4j или None в случае ошибки
        """
        try:
            # Извлечение необходимых данных
            concept_uri = concept_data.get("@id", "")
            concept_name = concept_data.get("label", "")
            
            if not concept_uri or not concept_name:
                logger.warning(f"Неполные данные концепта: {concept_data}")
                return None
            
            # Создание узла в Neo4j
            concept_node = self.create_node(
                "Concept",
                id=hashlib.md5(concept_uri.encode()).hexdigest(),
                uri=concept_uri,
                name=concept_name,
                language=self.lang,
                created_at=time.time()
            )
            
            logger.info(f"Концепт импортирован: {concept_name}")
            return concept_node
        except Exception as e:
            logger.error(f"Ошибка импорта концепта: {e}")
            return None
    
    def import_relation_to_neo4j(self, relation_data: Dict[str, Any]) -> Optional[Relationship]:
        """
        Импорт отношения в Neo4j
        
        Args:
            relation_data: Данные отношения
            
        Returns:
            Отношение Neo4j или None в случае ошибки
        """
        try:
            # Извлечение данных отношения
            relation_uri = relation_data.get("@id", "")
            relation_type = relation_data.get("rel", {}).get("@id", "").split("/")[-1]
            source_uri = relation_data.get("start", {}).get("@id", "")
            target_uri = relation_data.get("end", {}).get("@id", "")
            weight = relation_data.get("weight", 1.0)
            
            if not relation_uri or not relation_type or not source_uri or not target_uri:
                logger.warning(f"Неполные данные отношения: {relation_data}")
                return None
            
            # Получение или создание узлов концептов
            source_concept = self.get_concept(source_uri)
            target_concept = self.get_concept(target_uri)
            
            source_node = self.import_concept_to_neo4j(source_concept)
            target_node = self.import_concept_to_neo4j(target_concept)
            
            if not source_node or not target_node:
                logger.warning(f"Не удалось создать узлы для отношения {relation_uri}")
                return None
            
            # Создание отношения
            relationship = self.create_relationship(
                source_node,
                target_node,
                relation_type,
                id=hashlib.md5(relation_uri.encode()).hexdigest(),
                uri=relation_uri,
                weight=weight,
                created_at=time.time()
            )
            
            logger.info(f"Отношение импортировано: {source_node['name']}-[{relation_type}]->{target_node['name']}")
            return relationship
        except Exception as e:
            logger.error(f"Ошибка импорта отношения: {e}")
            return None
    
    async def import_concept_with_relations(self, concept: str, max_depth: int = 2):
        """
        Импорт концепта и его связей в Neo4j с ограничением глубины
        
        Args:
            concept: Концепт для импорта
            max_depth: Максимальная глубина обхода связей
        """
        # Начальная очередь концептов для обработки
        queue = [(concept, 0)]  # (концепт, глубина)
        processed = set()  # Множество обработанных концептов
        
        while queue:
            current_concept, depth = queue.pop(0)
            
            # Проверка на максимальную глубину
            if depth > max_depth:
                continue
            
            # Проверка на обработанный концепт
            if current_concept in processed:
                continue
            
            processed.add(current_concept)
            
            # Получение данных о концепте
            concept_data = self.get_concept(current_concept)
            if not concept_data:
                continue
            
            # Импорт концепта в Neo4j
            self.import_concept_to_neo4j(concept_data)
            
            # Получение связей концепта
            edges = concept_data.get("edges", [])
            for edge in edges:
                # Импорт отношения
                self.import_relation_to_neo4j(edge)
                
                # Добавление связанных концептов в очередь
                start_uri = edge.get("start", {}).get("@id", "")
                end_uri = edge.get("end", {}).get("@id", "")
                
                if start_uri and start_uri != current_concept and start_uri not in processed:
                    queue.append((start_uri, depth + 1))
                
                if end_uri and end_uri != current_concept and end_uri not in processed:
                    queue.append((end_uri, depth + 1))
            
            # Публикация события о прогрессе
            if self.nats_client and self.nats_client.connected:
                progress_event = {
                    "type": "concept_import_progress",
                    "concept": current_concept,
                    "depth": depth,
                    "max_depth": max_depth,
                    "processed_count": len(processed),
                    "timestamp": time.time()
                }
                
                await self.nats_client.publish(
                    "sasok.knowledge.conceptnet.progress",
                    json.dumps(progress_event).encode()
                )
            
            # Небольшая задержка, чтобы не перегружать API
            await asyncio.sleep(0.5)
        
        logger.info(f"Импорт концепта '{concept}' завершен. Обработано {len(processed)} концептов.")
        
        # Публикация события о завершении импорта
        if self.nats_client and self.nats_client.connected:
            completion_event = {
                "type": "concept_import_completed",
                "concept": concept,
                "max_depth": max_depth,
                "processed_count": len(processed),
                "timestamp": time.time()
            }
            
            await self.nats_client.publish(
                "sasok.knowledge.conceptnet.completed",
                json.dumps(completion_event).encode()
            )
    
    async def import_emotional_concepts(self, emotions: List[str], max_depth: int = 1):
        """
        Импорт эмоциональных концептов и их связей
        
        Args:
            emotions: Список эмоций для импорта
            max_depth: Максимальная глубина обхода связей
        """
        for emotion in emotions:
            logger.info(f"Импорт эмоционального концепта: {emotion}")
            await self.import_concept_with_relations(emotion, max_depth)
    
    async def find_emotional_path(self, source_concept: str, target_emotion: str, max_depth: int = 3) -> List[Dict]:
        """
        Поиск пути от концепта к эмоции в графе Neo4j
        
        Args:
            source_concept: Исходный концепт
            target_emotion: Целевая эмоция
            max_depth: Максимальная глубина поиска
            
        Returns:
            Список узлов и отношений на пути
        """
        # Выполнение Cypher-запроса для поиска пути
        query = f"""
        MATCH path = (source:Concept {{name: $source_name}})-[*1..{max_depth}]-(target:Concept {{name: $target_name}})
        RETURN path
        LIMIT 1
        """
        
        result = self.run_query(
            query,
            source_name=source_concept,
            target_name=target_emotion
        )
        
        if not result:
            logger.info(f"Путь от '{source_concept}' к '{target_emotion}' не найден")
            return []
        
        # Извлечение пути из результата
        path = result[0].get("path", None)
        if not path:
            return []
        
        # Преобразование пути в список узлов и отношений
        path_data = []
        # Обработка и форматирование пути...
        
        return path_data

# Пример использования
async def main():
    # Получение настроек из переменных окружения
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    
    # Создание процессора ConceptNet
    processor = ConceptNetProcessor(neo4j_uri, neo4j_user, neo4j_password)
    await processor.initialize()
    
    # Импорт базовых эмоциональных концептов
    basic_emotions = ["радость", "грусть", "злость", "страх", "удивление", "отвращение"]
    await processor.import_emotional_concepts(basic_emotions)
    
    # Закрытие соединений
    processor.close()

if __name__ == "__main__":
    asyncio.run(main())
