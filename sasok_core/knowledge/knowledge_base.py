"""
Модуль интеграции баз знаний для SASOK

Этот модуль обеспечивает интеграцию внешних баз знаний (ConceptNet, WordNet, Wikidata)
и эмоциональных датасетов (EmpatheticDialogues, GoEmotions) в семантическую сеть SASOK.
"""
import os
import logging
import json
import asyncio
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import importlib.util

import nltk
from nltk.corpus import wordnet as wn
import networkx as nx
from py2neo import Graph, Node, Relationship
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SASOK.KnowledgeBase")

# Путь к директории с данными
DATA_DIR = Path(os.getenv("DATA_DIR", "/home/sasok/Рабочий стол/blackboxai-1745739396945/data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/home/sasok/Рабочий стол/blackboxai-1745739396945/models"))

# Создание директорий, если не существуют
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Создание поддиректорий
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
EMOTION_DATASETS_DIR = DATA_DIR / "emotion_datasets"
KNOWLEDGE_DIR.mkdir(exist_ok=True)
EMOTION_DATASETS_DIR.mkdir(exist_ok=True)

class KnowledgeBase:
    """
    Базовый класс для работы с базами знаний и их интеграцией в Neo4j
    """
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """
        Инициализация базы знаний и подключения к Neo4j
        
        Args:
            neo4j_uri: URI для подключения к Neo4j
            neo4j_user: Имя пользователя Neo4j
            neo4j_password: Пароль пользователя Neo4j
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.graph = None
        
        # Подключение к Neo4j
        try:
            self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
            logger.info(f"Подключение к Neo4j успешно: {neo4j_uri}")
        except Exception as e:
            logger.error(f"Ошибка подключения к Neo4j: {e}")
    
    def create_index(self, label: str, property_name: str) -> None:
        """
        Создание индекса в Neo4j для ускорения запросов
        
        Args:
            label: Метка узла
            property_name: Имя свойства для индексации
        """
        if self.graph:
            try:
                self.graph.run(f"CREATE INDEX ON :{label}({property_name})")
                logger.info(f"Индекс создан: {label}.{property_name}")
            except Exception as e:
                # Если индекс уже существует, игнорируем ошибку
                logger.warning(f"Индекс уже существует или ошибка создания: {e}")
    
    def create_node(self, label: str, **properties) -> Optional[Node]:
        """
        Создание узла в Neo4j
        
        Args:
            label: Метка узла
            **properties: Свойства узла
            
        Returns:
            Node или None в случае ошибки
        """
        if not self.graph:
            logger.error("Нет подключения к Neo4j")
            return None
        
        try:
            # Поиск существующего узла
            query = f"MATCH (n:{label} {{id: $id}}) RETURN n"
            result = self.graph.run(query, id=properties.get("id")).data()
            
            if result:
                # Узел существует, обновляем свойства
                node = result[0]["n"]
                for key, value in properties.items():
                    node[key] = value
                self.graph.push(node)
                logger.debug(f"Узел обновлен: {label} с id {properties.get('id')}")
                return node
            else:
                # Создание нового узла
                node = Node(label, **properties)
                self.graph.create(node)
                logger.debug(f"Узел создан: {label} с id {properties.get('id')}")
                return node
        except Exception as e:
            logger.error(f"Ошибка создания/обновления узла: {e}")
            return None
    
    def create_relationship(self, 
                           source_node: Node, 
                           target_node: Node, 
                           rel_type: str, 
                           **properties) -> Optional[Relationship]:
        """
        Создание связи между узлами в Neo4j
        
        Args:
            source_node: Исходный узел
            target_node: Целевой узел
            rel_type: Тип связи
            **properties: Свойства связи
            
        Returns:
            Relationship или None в случае ошибки
        """
        if not self.graph:
            logger.error("Нет подключения к Neo4j")
            return None
        
        try:
            # Создание связи
            relationship = Relationship(source_node, rel_type, target_node, **properties)
            self.graph.create(relationship)
            logger.debug(f"Связь создана: ({source_node['id']})-[{rel_type}]->({target_node['id']})")
            return relationship
        except Exception as e:
            logger.error(f"Ошибка создания связи: {e}")
            return None
    
    def run_query(self, query: str, **params) -> List[Dict]:
        """
        Выполнение запроса к Neo4j
        
        Args:
            query: Cypher-запрос
            **params: Параметры запроса
            
        Returns:
            Список результатов
        """
        if not self.graph:
            logger.error("Нет подключения к Neo4j")
            return []
        
        try:
            result = self.graph.run(query, **params).data()
            return result
        except Exception as e:
            logger.error(f"Ошибка выполнения запроса: {e}")
            return []
    
    def close(self) -> None:
        """
        Закрытие подключения к Neo4j
        """
        if self.graph:
            try:
                # py2neo не требует явного закрытия соединения
                self.graph = None
                logger.info("Подключение к Neo4j закрыто")
            except Exception as e:
                logger.error(f"Ошибка закрытия подключения: {e}")
