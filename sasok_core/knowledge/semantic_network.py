"""
Модуль управления семантической сетью SASOK

Объединяет и координирует все источники знаний (ConceptNet, WordNet, 
Wikidata, эмоциональные датасеты) в единую семантическую сеть.
"""
import os
import logging
import json
import asyncio
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

from py2neo import Graph, Node, Relationship
import networkx as nx

from sasok_core.knowledge.knowledge_base import KnowledgeBase
from sasok_core.knowledge.concept_net import ConceptNetProcessor
from sasok_core.knowledge.wordnet_processor import WordNetProcessor
from sasok_core.knowledge.wikidata_processor import WikidataProcessor
from sasok_core.knowledge.emotion_datasets import EmotionDatasetsProcessor
from sasok_core.utils.nats_client import NatsClient

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SASOK.SemanticNetwork")

class SemanticNetwork:
    """
    Класс для управления и интеграции всех источников знаний в единую сеть
    """
    def __init__(self, 
                neo4j_uri: str, 
                neo4j_user: str, 
                neo4j_password: str, 
                lang: str = "ru"):
        """
        Инициализация семантической сети
        
        Args:
            neo4j_uri: URI для подключения к Neo4j
            neo4j_user: Имя пользователя Neo4j
            neo4j_password: Пароль пользователя Neo4j
            lang: Основной язык для работы с данными
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.lang = lang
        self.nats_client = None
        
        # Инициализация процессоров для разных источников знаний
        self.concept_net = ConceptNetProcessor(neo4j_uri, neo4j_user, neo4j_password, lang)
        self.wordnet = WordNetProcessor(neo4j_uri, neo4j_user, neo4j_password, lang)
        self.wikidata = WikidataProcessor(neo4j_uri, neo4j_user, neo4j_password, lang)
        self.emotion_datasets = EmotionDatasetsProcessor(neo4j_uri, neo4j_user, neo4j_password, lang)
        
        # Базовый доступ к Neo4j
        self.graph = None
        try:
            self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
            logger.info(f"Подключение к Neo4j успешно: {neo4j_uri}")
        except Exception as e:
            logger.error(f"Ошибка подключения к Neo4j: {e}")
    
    async def initialize(self):
        """
        Асинхронная инициализация всех компонентов
        """
        # Подключение к NATS
        nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
        self.nats_client = await NatsClient.get_instance(nats_url)
        
        # Инициализация всех процессоров
        await self.concept_net.initialize()
        await self.wordnet.initialize()
        await self.wikidata.initialize()
        await self.emotion_datasets.initialize()
        
        logger.info("Семантическая сеть инициализирована")
        
        # Подписка на события
        if self.nats_client and self.nats_client.connected:
            await self.subscribe_to_events()
    
    async def subscribe_to_events(self):
        """
        Подписка на события NATS для обработки запросов
        """
        # Подписка на запросы импорта знаний
        await self.nats_client.subscribe(
            "sasok.knowledge.import.request",
            self.handle_import_request
        )
        
        # Подписка на запросы поиска в семантической сети
        await self.nats_client.subscribe(
            "sasok.knowledge.search.request",
            self.handle_search_request
        )
        
        # Подписка на запросы эмоционального анализа концептов
        await self.nats_client.subscribe(
            "sasok.knowledge.emotion.request",
            self.handle_emotion_request
        )
        
        logger.info("Подписка на события NATS выполнена")
    
    async def handle_import_request(self, msg):
        """
        Обработка запроса на импорт знаний
        
        Args:
            msg: Сообщение NATS
        """
        try:
            # Распаковка данных запроса
            data = json.loads(msg.data.decode())
            
            source = data.get("source", "")
            params = data.get("params", {})
            
            logger.info(f"Получен запрос на импорт знаний из источника: {source}")
            
            # Обработка запроса в зависимости от источника
            if source == "conceptnet":
                # Импорт данных из ConceptNet
                max_depth = params.get("max_depth", 2)
                emotions = params.get("emotions", [])
                
                if emotions:
                    await self.concept_net.import_emotional_concepts(emotions, max_depth)
                else:
                    # Импорт концепта со связями
                    concept = params.get("concept", "")
                    if concept:
                        await self.concept_net.import_concept_with_relations(concept, max_depth)
            
            elif source == "wordnet":
                # Импорт данных из WordNet
                max_depth = params.get("max_depth", 2)
                
                # Импорт эмоциональной сети
                await self.wordnet.import_emotion_network(max_depth)
                
                # Экспорт эмоционального лексикона
                export_lexicon = params.get("export_lexicon", False)
                if export_lexicon:
                    output_file = params.get("output_file", None)
                    await self.wordnet.export_emotion_lexicon(output_file)
            
            elif source == "wikidata":
                # Импорт данных из Wikidata
                max_depth = params.get("max_depth", 1)
                
                # Импорт эмоциональных сущностей
                emotions = params.get("emotions", None)
                await self.wikidata.import_emotion_entities(emotions)
                
                # Импорт эмоциональной сети
                await self.wikidata.import_emotion_network(max_depth)
            
            elif source == "emotion_datasets":
                # Импорт эмоциональных датасетов
                await self.emotion_datasets.import_all_datasets()
                
                # Обучение модели
                train_model = params.get("train_model", False)
                if train_model:
                    model_name = params.get("model_name", "distilbert-base-uncased")
                    dataset_name = params.get("dataset_name", "combined")
                    output_dir = params.get("output_dir", None)
                    
                    await self.emotion_datasets.fine_tune_model(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        output_dir=output_dir
                    )
            
            elif source == "all":
                # Импорт данных из всех источников
                max_depth = params.get("max_depth", 1)
                
                # Запуск импорта в параллельных задачах
                tasks = [
                    self.concept_net.import_emotional_concepts(
                        ["радость", "грусть", "злость", "страх", "удивление", "отвращение"],
                        max_depth
                    ),
                    self.wordnet.import_emotion_network(max_depth),
                    self.wikidata.import_emotion_entities(),
                    self.emotion_datasets.import_all_datasets()
                ]
                
                await asyncio.gather(*tasks)
            
            else:
                logger.warning(f"Неизвестный источник знаний: {source}")
            
            # Публикация события о завершении импорта
            response = {
                "type": "import_completed",
                "source": source,
                "timestamp": time.time()
            }
            
            await self.nats_client.publish(
                "sasok.knowledge.import.response",
                json.dumps(response).encode()
            )
        
        except Exception as e:
            logger.error(f"Ошибка обработки запроса импорта: {e}")
            
            # Публикация события об ошибке
            error_response = {
                "type": "import_error",
                "source": data.get("source", "") if "data" in locals() else "unknown",
                "error": str(e),
                "timestamp": time.time()
            }
            
            await self.nats_client.publish(
                "sasok.knowledge.import.response",
                json.dumps(error_response).encode()
            )
    
    async def handle_search_request(self, msg):
        """
        Обработка запроса на поиск в семантической сети
        
        Args:
            msg: Сообщение NATS
        """
        try:
            # Распаковка данных запроса
            data = json.loads(msg.data.decode())
            
            query = data.get("query", "")
            sources = data.get("sources", ["all"])
            params = data.get("params", {})
            
            logger.info(f"Получен запрос на поиск: {query}")
            
            results = {}
            
            # Поиск в ConceptNet
            if "conceptnet" in sources or "all" in sources:
                if "concept" in params:
                    # Поиск по имени концепта
                    concept_results = self.concept_net.search_concept(params["concept"])
                else:
                    # Поиск по запросу
                    concept_results = self.concept_net.search_concept(query)
                
                results["conceptnet"] = concept_results
            
            # Поиск в WordNet
            if "wordnet" in sources or "all" in sources:
                if "word" in params:
                    # Поиск синсетов по слову
                    word_results = self.wordnet.get_synset_by_name(params["word"])
                    results["wordnet"] = [
                        self.wordnet.synset_to_dict(synset) for synset in word_results
                    ]
                else:
                    # Поиск по запросу
                    synsets = self.wordnet.get_synset_by_name(query)
                    results["wordnet"] = [
                        self.wordnet.synset_to_dict(synset) for synset in synsets
                    ]
            
            # Поиск в Wikidata
            if "wikidata" in sources or "all" in sources:
                if "entity" in params:
                    # Поиск сущности по имени
                    entity_results = self.wikidata.search_entity(params["entity"])
                else:
                    # Поиск по запросу
                    entity_results = self.wikidata.search_entity(query)
                
                results["wikidata"] = entity_results
            
            # Поиск эмоциональных путей
            if "emotion_paths" in params and params["emotion_paths"]:
                emotion = params.get("emotion", "радость")
                max_depth = params.get("max_depth", 3)
                
                # Поиск путей в разных источниках
                paths = {}
                
                if "conceptnet" in sources or "all" in sources:
                    concept_path = await self.concept_net.find_emotional_path(query, emotion, max_depth)
                    paths["conceptnet"] = concept_path
                
                if "wordnet" in sources or "all" in sources:
                    wordnet_path = await self.wordnet.find_emotion_path(query, emotion, max_depth)
                    paths["wordnet"] = wordnet_path
                
                if "wikidata" in sources or "all" in sources:
                    wikidata_path = await self.wikidata.find_emotion_path(query, emotion, max_depth)
                    paths["wikidata"] = wikidata_path
                
                results["emotion_paths"] = paths
            
            # Публикация результатов поиска
            response = {
                "type": "search_results",
                "query": query,
                "sources": sources,
                "results": results,
                "timestamp": time.time()
            }
            
            await self.nats_client.publish(
                "sasok.knowledge.search.response",
                json.dumps(response).encode()
            )
        
        except Exception as e:
            logger.error(f"Ошибка обработки запроса поиска: {e}")
            
            # Публикация события об ошибке
            error_response = {
                "type": "search_error",
                "query": data.get("query", "") if "data" in locals() else "unknown",
                "error": str(e),
                "timestamp": time.time()
            }
            
            await self.nats_client.publish(
                "sasok.knowledge.search.response",
                json.dumps(error_response).encode()
            )
    
    async def handle_emotion_request(self, msg):
        """
        Обработка запроса на эмоциональный анализ концептов
        
        Args:
            msg: Сообщение NATS
        """
        try:
            # Распаковка данных запроса
            data = json.loads(msg.data.decode())
            
            concept = data.get("concept", "")
            emotion_type = data.get("emotion_type", "basic")
            
            logger.info(f"Получен запрос на эмоциональный анализ концепта: {concept}")
            
            # Получение эмоционального профиля концепта
            emotion_profile = await self.analyze_concept_emotions(concept, emotion_type)
            
            # Публикация результатов анализа
            response = {
                "type": "emotion_analysis",
                "concept": concept,
                "emotion_type": emotion_type,
                "emotion_profile": emotion_profile,
                "timestamp": time.time()
            }
            
            await self.nats_client.publish(
                "sasok.knowledge.emotion.response",
                json.dumps(response).encode()
            )
        
        except Exception as e:
            logger.error(f"Ошибка обработки запроса эмоционального анализа: {e}")
            
            # Публикация события об ошибке
            error_response = {
                "type": "emotion_error",
                "concept": data.get("concept", "") if "data" in locals() else "unknown",
                "error": str(e),
                "timestamp": time.time()
            }
            
            await self.nats_client.publish(
                "sasok.knowledge.emotion.response",
                json.dumps(error_response).encode()
            )
    
    async def analyze_concept_emotions(self, concept: str, emotion_type: str = "basic") -> Dict[str, float]:
        """
        Анализ эмоционального профиля концепта
        
        Args:
            concept: Концепт для анализа
            emotion_type: Тип эмоций (basic, complex, all)
            
        Returns:
            Словарь эмоций с весами
        """
        # Базовые эмоции
        basic_emotions = ["радость", "грусть", "злость", "страх", "удивление", "отвращение"]
        
        # Расширенный набор эмоций
        complex_emotions = [
            "любовь", "ненависть", "тревога", "спокойствие", "восторг", "печаль",
            "гордость", "стыд", "вина", "зависть", "ревность", "благодарность",
            "презрение", "разочарование", "надежда", "облегчение", "сострадание"
        ]
        
        # Определение списка эмоций для анализа
        if emotion_type == "basic":
            emotions = basic_emotions
        elif emotion_type == "complex":
            emotions = complex_emotions
        else:  # "all"
            emotions = basic_emotions + complex_emotions
        
        # Объединенный профиль из нескольких источников
        emotion_profile = {emotion: 0.0 for emotion in emotions}
        
        try:
            # Cypher-запрос для получения всех эмоциональных путей
            query = """
            // Поиск в ConceptNet
            MATCH path1 = (c1:Concept)-[*1..3]-(e1:Concept)
            WHERE c1.name CONTAINS $concept AND e1.name IN $emotions
            
            // Поиск в WordNet
            OPTIONAL MATCH path2 = (w:WordNetWord)-[:IN_SYNSET]->(s:WordNetSynset)-[*1..3]-(es:WordNetSynset)<-[:IN_SYNSET]-(ew:WordNetWord)
            WHERE w.name CONTAINS $concept AND ew.name IN $emotions
            
            // Поиск в Wikidata
            OPTIONAL MATCH path3 = (wd:WikidataEntity)-[*1..3]-(ewd:WikidataEntity:Emotion)
            WHERE wd.label CONTAINS $concept AND ewd.label IN $emotions
            
            // Возвращаем все пути и эмоции
            RETURN 
                [e IN $emotions | {
                    emotion: e,
                    conceptnet_paths: size([p IN collect(path1) WHERE nodes(p)[-1].name = e]),
                    wordnet_paths: size([p IN collect(path2) WHERE nodes(p)[-1].name = e]),
                    wikidata_paths: size([p IN collect(path3) WHERE nodes(p)[-1].label = e])
                }] AS results
            """
            
            result = self.graph.run(query, concept=concept, emotions=emotions).data()
            
            if result and result[0]["results"]:
                # Обработка результатов
                for item in result[0]["results"]:
                    emotion = item["emotion"]
                    
                    # Подсчет общего количества путей
                    total_paths = (
                        item["conceptnet_paths"] + 
                        item["wordnet_paths"] + 
                        item["wikidata_paths"]
                    )
                    
                    # Если есть пути, вычисляем вес
                    if total_paths > 0:
                        emotion_profile[emotion] = min(1.0, total_paths / 10.0)  # Нормализация, максимум 1.0
            
            # Нормализация весов, чтобы сумма была 1.0
            total_weight = sum(emotion_profile.values())
            if total_weight > 0:
                emotion_profile = {emotion: weight / total_weight for emotion, weight in emotion_profile.items()}
        
        except Exception as e:
            logger.error(f"Ошибка при анализе эмоций концепта: {e}")
        
        return emotion_profile
    
    async def build_emotional_profile(self, text: str) -> Dict[str, float]:
        """
        Построение эмоционального профиля текста на основе семантической сети
        
        Args:
            text: Текст для анализа
            
        Returns:
            Словарь эмоций с весами
        """
        # Анализ текста и выделение ключевых концептов
        # Здесь должна быть реализация NLP-анализа текста
        
        # Пока используем упрощенный подход - разбиваем на слова
        import re
        from collections import Counter
        
        # Токенизация и очистка текста
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Фильтрация стоп-слов (упрощенно)
        stop_words = {"и", "в", "на", "с", "по", "для", "не", "от", "к", "у", "из", "а", "о", "что", "это"}
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Подсчет частоты слов
        word_counts = Counter(filtered_words)
        
        # Получение наиболее частых слов
        top_words = [word for word, count in word_counts.most_common(10)]
        
        # Анализ эмоций для каждого слова
        emotion_profiles = []
        
        for word in top_words:
            profile = await self.analyze_concept_emotions(word)
            emotion_profiles.append(profile)
        
        # Объединение эмоциональных профилей
        combined_profile = {}
        
        if emotion_profiles:
            # Сначала собираем все эмоции
            all_emotions = set()
            for profile in emotion_profiles:
                all_emotions.update(profile.keys())
            
            # Вычисляем средние веса
            for emotion in all_emotions:
                weights = [profile.get(emotion, 0.0) for profile in emotion_profiles]
                combined_profile[emotion] = sum(weights) / len(emotion_profiles)
        
        return combined_profile
    
    def close(self):
        """
        Закрытие всех соединений
        """
        # Закрытие соединений с Neo4j
        self.concept_net.close()
        self.wordnet.close()
        self.wikidata.close()
        self.emotion_datasets.close()
        
        if self.graph:
            self.graph = None
        
        logger.info("Соединения семантической сети закрыты")

# Пример использования
async def main():
    # Получение настроек из переменных окружения
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    
    # Создание семантической сети
    network = SemanticNetwork(neo4j_uri, neo4j_user, neo4j_password)
    await network.initialize()
    
    # Импорт базовых эмоциональных концептов
    import_request = {
        "source": "conceptnet",
        "params": {
            "emotions": ["радость", "грусть", "злость", "страх", "удивление", "отвращение"],
            "max_depth": 1
        }
    }
    
    # Эмулируем получение сообщения
    class MockMessage:
        def __init__(self, data):
            self.data = data.encode()
    
    msg = MockMessage(json.dumps(import_request))
    await network.handle_import_request(msg)
    
    # Поиск эмоциональной информации о концепте
    search_request = {
        "query": "любовь",
        "sources": ["all"],
        "params": {
            "emotion_paths": True,
            "emotion": "радость"
        }
    }
    
    msg = MockMessage(json.dumps(search_request))
    await network.handle_search_request(msg)
    
    # Анализ эмоций концепта
    emotion_request = {
        "concept": "любовь",
        "emotion_type": "all"
    }
    
    msg = MockMessage(json.dumps(emotion_request))
    await network.handle_emotion_request(msg)
    
    # Закрытие соединений
    network.close()

if __name__ == "__main__":
    asyncio.run(main())
