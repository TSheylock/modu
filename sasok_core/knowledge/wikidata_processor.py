"""
Модуль интеграции Wikidata в семантическую сеть SASOK

Загружает и обрабатывает данные из Wikidata, интегрируя их в Neo4j граф для
расширения семантической сети и обогащения эмоционального анализа.
"""
import os
import logging
import json
import asyncio
import time
import hashlib
import requests
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

from py2neo import Node, Relationship
from tqdm import tqdm

from sasok_core.knowledge.knowledge_base import KnowledgeBase
from sasok_core.utils.nats_client import NatsClient

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SASOK.Wikidata")

# Путь к файлам данных
DATA_DIR = Path(os.getenv("DATA_DIR", "/home/sasok/Рабочий стол/blackboxai-1745739396945/data"))
WIKIDATA_DIR = DATA_DIR / "knowledge" / "wikidata"
WIKIDATA_DIR.mkdir(parents=True, exist_ok=True)

# URL для API Wikidata
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# ID эмоций в Wikidata
EMOTION_QID_MAP = {
    "радость": "Q9415",
    "счастье": "Q8",
    "грусть": "Q8342",
    "печаль": "Q8342",
    "злость": "Q49049",
    "гнев": "Q49049",
    "страх": "Q9415",
    "удивление": "Q1546889",
    "отвращение": "Q744368",
    "презрение": "Q868467",
    "любовь": "Q316",
    "ненависть": "Q816377",
    "тревога": "Q10617",
    "спокойствие": "Q12800",
    "восторг": "Q1046155"
}

class WikidataProcessor(KnowledgeBase):
    """
    Класс для обработки и интеграции данных из Wikidata
    """
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, lang: str = "ru"):
        """
        Инициализация процессора Wikidata
        
        Args:
            neo4j_uri: URI для подключения к Neo4j
            neo4j_user: Имя пользователя Neo4j
            neo4j_password: Пароль пользователя Neo4j
            lang: Код языка для получения меток (ru для русского, en для английского)
        """
        super().__init__(neo4j_uri, neo4j_user, neo4j_password)
        self.lang = lang
        self.nats_client = None
        self.entity_cache = {}
        
        # Создание индексов в Neo4j
        self.create_index("WikidataEntity", "id")
        self.create_index("WikidataEntity", "qid")
        self.create_index("WikidataEntity", "label")
    
    async def initialize(self):
        """
        Асинхронная инициализация, включая подключение к NATS
        """
        # Подключение к NATS
        nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
        self.nats_client = await NatsClient.get_instance(nats_url)
        logger.info("WikidataProcessor инициализирован и подключен к NATS")
    
    def get_entity_by_qid(self, qid: str) -> Dict[str, Any]:
        """
        Получение сущности Wikidata по QID
        
        Args:
            qid: QID сущности Wikidata (например, Q9415)
            
        Returns:
            Информация о сущности
        """
        # Проверка кэша
        if qid in self.entity_cache:
            return self.entity_cache[qid]
        
        # Запрос к API
        try:
            params = {
                "action": "wbgetentities",
                "ids": qid,
                "languages": self.lang,
                "format": "json"
            }
            
            response = requests.get(WIKIDATA_API, params=params)
            if response.status_code == 200:
                data = response.json()
                
                # Извлечение данных
                entity_data = data.get("entities", {}).get(qid, {})
                
                # Кэширование результата
                self.entity_cache[qid] = entity_data
                
                return entity_data
            else:
                logger.warning(f"Ошибка получения сущности {qid}: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Ошибка API Wikidata: {e}")
            return {}
    
    def search_entity(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Поиск сущностей Wikidata
        
        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов
            
        Returns:
            Список найденных сущностей
        """
        # Кэш-ключ для запроса
        cache_key = f"search:{query}:{limit}"
        if cache_key in self.entity_cache:
            return self.entity_cache[cache_key]
        
        # Запрос к API
        try:
            params = {
                "action": "wbsearchentities",
                "search": query,
                "language": self.lang,
                "format": "json",
                "limit": limit
            }
            
            response = requests.get(WIKIDATA_API, params=params)
            if response.status_code == 200:
                data = response.json()
                
                # Извлечение результатов
                search_results = data.get("search", [])
                
                # Кэширование результата
                self.entity_cache[cache_key] = search_results
                
                return search_results
            else:
                logger.warning(f"Ошибка поиска сущностей '{query}': {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Ошибка API Wikidata: {e}")
            return []
    
    def run_sparql_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Выполнение SPARQL-запроса к Wikidata
        
        Args:
            query: SPARQL-запрос
            
        Returns:
            Результаты запроса
        """
        # Запрос к SPARQL-endpoint
        try:
            params = {
                "query": query,
                "format": "json"
            }
            
            headers = {
                "Accept": "application/sparql-results+json",
                "User-Agent": "SASOK/1.0 (https://saske.xyz; sasok@example.com)"
            }
            
            response = requests.get(WIKIDATA_SPARQL_ENDPOINT, params=params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                
                # Извлечение результатов
                results = data.get("results", {}).get("bindings", [])
                
                return results
            else:
                logger.warning(f"Ошибка выполнения SPARQL-запроса: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Ошибка SPARQL-endpoint Wikidata: {e}")
            return []
    
    def get_related_entities(self, qid: str) -> List[Dict[str, Any]]:
        """
        Получение связанных сущностей для заданного QID
        
        Args:
            qid: QID сущности Wikidata
            
        Returns:
            Список связанных сущностей с информацией о связях
        """
        # Формирование SPARQL-запроса
        query = """
        SELECT ?item ?itemLabel ?property ?propertyLabel
        WHERE {
          # Связи исходящие от сущности
          { wd:%s ?property ?item . }
          UNION
          # Связи входящие в сущность
          { ?item ?property wd:%s . }
          
          # Получение меток
          SERVICE wikibase:label { 
            bd:serviceParam wikibase:language "%s" . 
            ?item rdfs:label ?itemLabel .
            ?property rdfs:label ?propertyLabel .
          }
          
          # Фильтрация служебных свойств
          FILTER(REGEX(STR(?property), "^http://www.wikidata.org/prop/direct/"))
        }
        LIMIT 500
        """ % (qid, qid, self.lang)
        
        return self.run_sparql_query(query)
    
    def import_entity_to_neo4j(self, entity_data: Dict[str, Any]) -> Optional[Node]:
        """
        Импорт сущности Wikidata в Neo4j
        
        Args:
            entity_data: Данные сущности
            
        Returns:
            Узел Neo4j или None в случае ошибки
        """
        try:
            # Извлечение необходимых данных
            qid = entity_data.get("id", "")
            labels = entity_data.get("labels", {})
            descriptions = entity_data.get("descriptions", {})
            
            # Получение метки на нужном языке
            label = labels.get(self.lang, {}).get("value", qid)
            description = descriptions.get(self.lang, {}).get("value", "")
            
            if not qid:
                logger.warning(f"Неполные данные сущности: отсутствует QID")
                return None
            
            # Создание узла в Neo4j
            entity_node = self.create_node(
                "WikidataEntity",
                id=hashlib.md5(qid.encode()).hexdigest(),
                qid=qid,
                label=label,
                description=description,
                language=self.lang,
                created_at=time.time()
            )
            
            logger.info(f"Сущность импортирована: {label} ({qid})")
            return entity_node
        except Exception as e:
            logger.error(f"Ошибка импорта сущности: {e}")
            return None
    
    def import_relation_to_neo4j(self, source_entity: Node, target_entity: Node, relation_data: Dict[str, Any]) -> Optional[Relationship]:
        """
        Импорт отношения в Neo4j
        
        Args:
            source_entity: Исходная сущность
            target_entity: Целевая сущность
            relation_data: Данные отношения
            
        Returns:
            Отношение Neo4j или None в случае ошибки
        """
        try:
            # Извлечение данных отношения
            property_uri = relation_data.get("property", {}).get("value", "")
            property_label = relation_data.get("propertyLabel", {}).get("value", "")
            
            if not property_uri or not property_label:
                logger.warning(f"Неполные данные отношения")
                return None
            
            # Извлечение идентификатора свойства
            property_id = property_uri.split("/")[-1]
            
            # Создание отношения
            relationship = self.create_relationship(
                source_entity,
                target_entity,
                property_id,
                label=property_label,
                uri=property_uri,
                created_at=time.time()
            )
            
            logger.info(f"Отношение импортировано: {source_entity['label']}-[{property_label}]->{target_entity['label']}")
            return relationship
        except Exception as e:
            logger.error(f"Ошибка импорта отношения: {e}")
            return None
    
    async def import_emotion_entities(self, emotions: Dict[str, str] = None):
        """
        Импорт сущностей эмоций из Wikidata
        
        Args:
            emotions: Словарь соответствия названий эмоций и их QID
        """
        if emotions is None:
            emotions = EMOTION_QID_MAP
        
        imported_emotions = []
        
        for emotion_name, qid in emotions.items():
            logger.info(f"Импорт эмоции '{emotion_name}' (QID: {qid})")
            
            # Получение данных сущности
            entity_data = self.get_entity_by_qid(qid)
            
            if not entity_data:
                logger.warning(f"Не удалось получить данные для QID {qid}")
                continue
            
            # Импорт сущности в Neo4j
            entity_node = self.import_entity_to_neo4j(entity_data)
            
            if entity_node:
                # Добавление специальной метки для эмоций
                entity_node.add_label("Emotion")
                self.graph.push(entity_node)
                
                imported_emotions.append({
                    "name": emotion_name,
                    "qid": qid,
                    "label": entity_node["label"]
                })
            
            # Небольшая задержка, чтобы не перегружать API
            await asyncio.sleep(0.5)
        
        logger.info(f"Импортировано {len(imported_emotions)} эмоций")
        
        # Публикация события о завершении импорта
        if self.nats_client and self.nats_client.connected:
            event = {
                "type": "wikidata_emotions_imported",
                "emotions": imported_emotions,
                "count": len(imported_emotions),
                "timestamp": time.time()
            }
            
            await self.nats_client.publish(
                "sasok.knowledge.wikidata.emotions",
                json.dumps(event).encode()
            )
    
    async def import_emotion_network(self, max_depth: int = 2):
        """
        Импорт сети эмоциональных сущностей с ограничением глубины
        
        Args:
            max_depth: Максимальная глубина обхода связей
        """
        # Импорт базовых эмоций
        await self.import_emotion_entities()
        
        # Получение импортированных эмоций из Neo4j
        query = """
        MATCH (e:WikidataEntity:Emotion)
        RETURN e.qid AS qid, e.label AS label
        """
        
        emotion_entities = self.run_query(query)
        
        if not emotion_entities:
            logger.warning("Не найдены импортированные эмоции")
            return
        
        # Начальная очередь сущностей для обработки
        queue = [(entity["qid"], 0) for entity in emotion_entities]  # (QID, глубина)
        processed = set()  # Множество обработанных сущностей (по QID)
        
        total_entities = 0
        total_relations = 0
        
        while queue:
            current_qid, depth = queue.pop(0)
            
            # Проверка на максимальную глубину
            if depth > max_depth:
                continue
            
            # Проверка на обработанную сущность
            if current_qid in processed:
                continue
            
            processed.add(current_qid)
            
            # Получение данных сущности
            entity_data = self.get_entity_by_qid(current_qid)
            
            if not entity_data:
                logger.warning(f"Не удалось получить данные для QID {current_qid}")
                continue
            
            # Импорт сущности в Neo4j
            entity_node = self.import_entity_to_neo4j(entity_data)
            
            if not entity_node:
                continue
            
            total_entities += 1
            
            # Получение связанных сущностей
            related_entities = self.get_related_entities(current_qid)
            
            for related_entity in related_entities:
                # Извлечение данных связанной сущности
                related_qid = related_entity.get("item", {}).get("value", "").split("/")[-1]
                
                if not related_qid or not related_qid.startswith("Q"):
                    continue
                
                # Получение данных связанной сущности
                related_data = self.get_entity_by_qid(related_qid)
                
                if not related_data:
                    continue
                
                # Импорт связанной сущности
                related_node = self.import_entity_to_neo4j(related_data)
                
                if not related_node:
                    continue
                
                # Импорт отношения
                self.import_relation_to_neo4j(entity_node, related_node, related_entity)
                
                total_relations += 1
                
                # Добавление связанной сущности в очередь
                if related_qid not in processed:
                    queue.append((related_qid, depth + 1))
            
            # Публикация события о прогрессе
            if self.nats_client and self.nats_client.connected and total_entities % 10 == 0:
                progress_event = {
                    "type": "wikidata_network_progress",
                    "entity": current_qid,
                    "depth": depth,
                    "max_depth": max_depth,
                    "processed_count": len(processed),
                    "total_entities": total_entities,
                    "total_relations": total_relations,
                    "timestamp": time.time()
                }
                
                await self.nats_client.publish(
                    "sasok.knowledge.wikidata.progress",
                    json.dumps(progress_event).encode()
                )
            
            # Небольшая задержка, чтобы не перегружать API
            await asyncio.sleep(0.5)
        
        logger.info(f"Импорт эмоциональной сети Wikidata завершен. Обработано {total_entities} сущностей и {total_relations} отношений.")
        
        # Публикация события о завершении импорта
        if self.nats_client and self.nats_client.connected:
            completion_event = {
                "type": "wikidata_network_completed",
                "max_depth": max_depth,
                "processed_count": len(processed),
                "total_entities": total_entities,
                "total_relations": total_relations,
                "timestamp": time.time()
            }
            
            await self.nats_client.publish(
                "sasok.knowledge.wikidata.completed",
                json.dumps(completion_event).encode()
            )
    
    async def find_emotion_path(self, concept: str, emotion: str, max_depth: int = 3) -> List[Dict]:
        """
        Поиск пути от концепта к эмоции в графе Wikidata
        
        Args:
            concept: Исходное понятие (текст для поиска)
            emotion: Целевая эмоция (текст для поиска)
            max_depth: Максимальная глубина поиска
            
        Returns:
            Список узлов и отношений на пути
        """
        # Поиск сущностей
        concept_entities = self.search_entity(concept)
        emotion_entities = self.search_entity(emotion)
        
        if not concept_entities or not emotion_entities:
            logger.warning(f"Не найдены сущности для концепта '{concept}' или эмоции '{emotion}'")
            return []
        
        # Получение QID
        concept_qids = [entity.get("id") for entity in concept_entities]
        emotion_qids = [entity.get("id") for entity in emotion_entities]
        
        # Поиск кратчайшего пути в Neo4j
        query = f"""
        MATCH (source:WikidataEntity)
        WHERE source.qid IN $concept_qids
        MATCH (target:WikidataEntity)
        WHERE target.qid IN $emotion_qids
        MATCH path = shortestPath((source)-[*1..{max_depth}]-(target))
        RETURN path
        LIMIT 1
        """
        
        result = self.run_query(
            query,
            concept_qids=concept_qids,
            emotion_qids=emotion_qids
        )
        
        if not result:
            logger.info(f"Путь от '{concept}' к '{emotion}' не найден")
            return []
        
        # Извлечение пути из результата
        path = result[0].get("path", None)
        if not path:
            return []
        
        # Преобразование пути в список узлов и отношений
        path_data = []
        # Обработка пути...
        
        return path_data

# Пример использования
async def main():
    # Получение настроек из переменных окружения
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    
    # Создание процессора Wikidata
    processor = WikidataProcessor(neo4j_uri, neo4j_user, neo4j_password)
    await processor.initialize()
    
    # Импорт эмоциональных сущностей
    await processor.import_emotion_entities()
    
    # Импорт эмоциональной сети
    await processor.import_emotion_network(max_depth=1)
    
    # Закрытие соединений
    processor.close()

if __name__ == "__main__":
    asyncio.run(main())
