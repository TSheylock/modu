"""
Модуль интеграции WordNet в семантическую сеть SASOK

Загружает и обрабатывает данные из WordNet, интегрируя их в Neo4j граф для
расширения семантической сети и обогащения эмоционального анализа.
"""
import os
import logging
import json
import asyncio
import time
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

import nltk
from nltk.corpus import wordnet as wn
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
logger = logging.getLogger("SASOK.WordNet")

# Путь к файлам данных
DATA_DIR = Path(os.getenv("DATA_DIR", "/home/sasok/Рабочий стол/blackboxai-1745739396945/data"))
WORDNET_DIR = DATA_DIR / "knowledge" / "wordnet"
WORDNET_DIR.mkdir(parents=True, exist_ok=True)

# Типы отношений в WordNet
WORDNET_RELATIONS = {
    "hypernym": "IS_A",
    "hyponym": "HAS_TYPE",
    "member_holonym": "HAS_MEMBER",
    "member_meronym": "MEMBER_OF",
    "part_holonym": "HAS_PART",
    "part_meronym": "PART_OF",
    "substance_holonym": "HAS_SUBSTANCE",
    "substance_meronym": "SUBSTANCE_OF",
    "entailment": "ENTAILS",
    "also_see": "ALSO_SEE",
    "similar_to": "SIMILAR_TO",
    "attribute": "HAS_ATTRIBUTE",
    "cause": "CAUSES",
    "verb_group": "IN_VERB_GROUP",
    "derivationally_related": "DERIVATIONALLY_RELATED",
    "antonym": "ANTONYM_OF"
}

# Части речи в WordNet
POS_MAPPING = {
    wn.NOUN: "NOUN",
    wn.VERB: "VERB",
    wn.ADJ: "ADJECTIVE",
    wn.ADV: "ADVERB"
}

class WordNetProcessor(KnowledgeBase):
    """
    Класс для обработки и интеграции данных из WordNet
    """
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, lang: str = "rus"):
        """
        Инициализация процессора WordNet
        
        Args:
            neo4j_uri: URI для подключения к Neo4j
            neo4j_user: Имя пользователя Neo4j
            neo4j_password: Пароль пользователя Neo4j
            lang: Код языка WordNet (rus для русского, eng для английского)
        """
        super().__init__(neo4j_uri, neo4j_user, neo4j_password)
        self.lang = lang
        self.nats_client = None
        self.synset_cache = {}
        
        # Создание индексов в Neo4j
        self.create_index("WordNetSynset", "id")
        self.create_index("WordNetSynset", "name")
        self.create_index("WordNetWord", "name")
    
    async def initialize(self):
        """
        Асинхронная инициализация, включая подключение к NATS и загрузку NLTK
        """
        # Загрузка необходимых ресурсов NLTK
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("Загрузка WordNet...")
            nltk.download('wordnet')
            
            # Для русского WordNet может потребоваться дополнительная загрузка
            if self.lang == "rus":
                try:
                    nltk.data.find('corpora/omw-1.4')
                except LookupError:
                    logger.info("Загрузка Open Multilingual WordNet...")
                    nltk.download('omw-1.4')
        
        # Подключение к NATS
        nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
        self.nats_client = await NatsClient.get_instance(nats_url)
        logger.info("WordNetProcessor инициализирован и подключен к NATS")
    
    def get_synset_by_name(self, name: str, pos: Optional[str] = None) -> List[Any]:
        """
        Получение синсетов WordNet по имени
        
        Args:
            name: Имя (слово) для поиска
            pos: Часть речи (опционально)
            
        Returns:
            Список синсетов
        """
        # Преобразование POS в формат WordNet
        wordnet_pos = None
        if pos:
            for wn_pos, pos_name in POS_MAPPING.items():
                if pos_name.lower() == pos.lower():
                    wordnet_pos = wn_pos
                    break
        
        # Получение синсетов
        synsets = wn.synsets(name, pos=wordnet_pos, lang=self.lang)
        return synsets
    
    def get_synset_by_id(self, synset_id: str) -> Optional[Any]:
        """
        Получение синсета WordNet по ID
        
        Args:
            synset_id: ID синсета в формате WordNet
            
        Returns:
            Синсет или None
        """
        try:
            return wn.synset(synset_id)
        except:
            logger.warning(f"Синсет с ID {synset_id} не найден")
            return None
    
    def get_emotion_synsets(self) -> List[Any]:
        """
        Получение синсетов, связанных с эмоциями
        
        Returns:
            Список синсетов эмоций
        """
        # Базовые эмоции
        emotion_words = [
            "радость", "счастье", "грусть", "печаль", "злость", "гнев", 
            "страх", "испуг", "удивление", "отвращение", "презрение",
            "любовь", "ненависть", "тревога", "спокойствие", "восторг"
        ]
        
        # Получение синсетов для каждого слова
        emotion_synsets = []
        for word in emotion_words:
            synsets = self.get_synset_by_name(word)
            emotion_synsets.extend(synsets)
        
        return emotion_synsets
    
    def synset_to_dict(self, synset: Any) -> Dict[str, Any]:
        """
        Преобразование синсета в словарь
        
        Args:
            synset: Синсет WordNet
            
        Returns:
            Словарь с данными синсета
        """
        # Извлечение данных синсета
        synset_dict = {
            "id": synset.name(),
            "name": synset.name().split('.')[0],
            "pos": POS_MAPPING.get(synset.pos(), "UNKNOWN"),
            "definition": synset.definition(),
            "examples": synset.examples(),
            "lemmas": [lemma.name() for lemma in synset.lemmas(lang=self.lang)],
            "hyponyms": [hypo.name() for hypo in synset.hyponyms()],
            "hypernyms": [hyper.name() for hyper in synset.hypernyms()]
        }
        
        return synset_dict
    
    def import_synset_to_neo4j(self, synset: Any) -> Optional[Node]:
        """
        Импорт синсета в Neo4j
        
        Args:
            synset: Синсет WordNet
            
        Returns:
            Узел Neo4j или None в случае ошибки
        """
        try:
            # Преобразование в словарь
            synset_data = self.synset_to_dict(synset)
            
            # Создание узла в Neo4j
            synset_node = self.create_node(
                "WordNetSynset",
                id=synset_data["id"],
                name=synset_data["name"],
                pos=synset_data["pos"],
                definition=synset_data["definition"],
                examples=json.dumps(synset_data["examples"]),
                lemmas=json.dumps(synset_data["lemmas"]),
                created_at=time.time()
            )
            
            # Создание узлов для лемм (слов)
            for lemma_name in synset_data["lemmas"]:
                lemma_node = self.create_node(
                    "WordNetWord",
                    id=hashlib.md5(f"{lemma_name}_{synset_data['pos']}".encode()).hexdigest(),
                    name=lemma_name,
                    pos=synset_data["pos"],
                    created_at=time.time()
                )
                
                # Создание связи между словом и синсетом
                if lemma_node:
                    self.create_relationship(
                        lemma_node,
                        synset_node,
                        "IN_SYNSET",
                        weight=1.0
                    )
            
            logger.info(f"Синсет импортирован: {synset_data['name']} ({synset_data['id']})")
            return synset_node
        except Exception as e:
            logger.error(f"Ошибка импорта синсета: {e}")
            return None
    
    def import_synset_relations_to_neo4j(self, synset: Any, synset_node: Node) -> int:
        """
        Импорт отношений синсета в Neo4j
        
        Args:
            synset: Синсет WordNet
            synset_node: Узел синсета в Neo4j
            
        Returns:
            Количество импортированных отношений
        """
        relation_count = 0
        
        # Обработка всех типов отношений
        for relation_type, neo4j_rel in WORDNET_RELATIONS.items():
            try:
                # Получение метода для доступа к отношению
                relation_method = getattr(synset, relation_type + 's', None)
                if not relation_method:
                    continue
                
                # Получение связанных синсетов
                related_synsets = relation_method()
                
                for related_synset in related_synsets:
                    # Импорт связанного синсета
                    related_node = self.import_synset_to_neo4j(related_synset)
                    
                    if related_node:
                        # Создание отношения
                        self.create_relationship(
                            synset_node,
                            related_node,
                            neo4j_rel,
                            weight=1.0
                        )
                        
                        relation_count += 1
            except Exception as e:
                logger.warning(f"Ошибка импорта отношения {relation_type}: {e}")
        
        return relation_count
    
    async def import_emotion_network(self, max_depth: int = 2):
        """
        Импорт сети эмоциональных синсетов с ограничением глубины
        
        Args:
            max_depth: Максимальная глубина обхода связей
        """
        # Получение эмоциональных синсетов
        emotion_synsets = self.get_emotion_synsets()
        
        # Начальная очередь синсетов для обработки
        queue = [(synset, 0) for synset in emotion_synsets]  # (синсет, глубина)
        processed = set()  # Множество обработанных синсетов (по ID)
        
        total_synsets = 0
        total_relations = 0
        
        while queue:
            current_synset, depth = queue.pop(0)
            
            # Проверка на максимальную глубину
            if depth > max_depth:
                continue
            
            # Проверка на обработанный синсет
            synset_id = current_synset.name()
            if synset_id in processed:
                continue
            
            processed.add(synset_id)
            
            # Импорт синсета в Neo4j
            synset_node = self.import_synset_to_neo4j(current_synset)
            if not synset_node:
                continue
            
            total_synsets += 1
            
            # Импорт отношений
            relation_count = self.import_synset_relations_to_neo4j(current_synset, synset_node)
            total_relations += relation_count
            
            # Добавление связанных синсетов в очередь
            for relation_type in WORDNET_RELATIONS.keys():
                try:
                    relation_method = getattr(current_synset, relation_type + 's', None)
                    if not relation_method:
                        continue
                    
                    related_synsets = relation_method()
                    for related_synset in related_synsets:
                        if related_synset.name() not in processed:
                            queue.append((related_synset, depth + 1))
                except Exception as e:
                    logger.warning(f"Ошибка получения связанных синсетов {relation_type}: {e}")
            
            # Публикация события о прогрессе
            if self.nats_client and self.nats_client.connected and total_synsets % 10 == 0:
                progress_event = {
                    "type": "wordnet_import_progress",
                    "synset": synset_id,
                    "depth": depth,
                    "max_depth": max_depth,
                    "processed_count": len(processed),
                    "total_synsets": total_synsets,
                    "total_relations": total_relations,
                    "timestamp": time.time()
                }
                
                await self.nats_client.publish(
                    "sasok.knowledge.wordnet.progress",
                    json.dumps(progress_event).encode()
                )
            
            # Небольшая задержка
            await asyncio.sleep(0.05)
        
        logger.info(f"Импорт эмоциональной сети WordNet завершен. Обработано {total_synsets} синсетов и {total_relations} отношений.")
        
        # Публикация события о завершении импорта
        if self.nats_client and self.nats_client.connected:
            completion_event = {
                "type": "wordnet_import_completed",
                "max_depth": max_depth,
                "processed_count": len(processed),
                "total_synsets": total_synsets,
                "total_relations": total_relations,
                "timestamp": time.time()
            }
            
            await self.nats_client.publish(
                "sasok.knowledge.wordnet.completed",
                json.dumps(completion_event).encode()
            )
    
    async def find_emotion_path(self, word: str, emotion: str, max_depth: int = 4) -> List[Dict]:
        """
        Поиск пути от слова к эмоции в графе WordNet
        
        Args:
            word: Исходное слово
            emotion: Целевая эмоция
            max_depth: Максимальная глубина поиска
            
        Returns:
            Список узлов и отношений на пути
        """
        # Преобразование слов в синсеты
        word_synsets = self.get_synset_by_name(word)
        emotion_synsets = self.get_synset_by_name(emotion)
        
        if not word_synsets or not emotion_synsets:
            logger.warning(f"Не найдены синсеты для слова '{word}' или эмоции '{emotion}'")
            return []
        
        # Поиск кратчайшего пути в Neo4j
        word_ids = [synset.name() for synset in word_synsets]
        emotion_ids = [synset.name() for synset in emotion_synsets]
        
        # Формирование запроса с множественными исходными и целевыми узлами
        query = f"""
        MATCH (source:WordNetSynset)
        WHERE source.id IN $word_ids
        MATCH (target:WordNetSynset)
        WHERE target.id IN $emotion_ids
        MATCH path = shortestPath((source)-[*1..{max_depth}]-(target))
        RETURN path
        LIMIT 1
        """
        
        result = self.run_query(
            query,
            word_ids=word_ids,
            emotion_ids=emotion_ids
        )
        
        if not result:
            logger.info(f"Путь от '{word}' к '{emotion}' не найден")
            return []
        
        # Извлечение пути из результата
        path = result[0].get("path", None)
        if not path:
            return []
        
        # Преобразование пути в список узлов и отношений
        path_data = []
        # Обработка пути...
        
        return path_data
    
    def get_emotion_lexicon(self) -> Dict[str, Dict[str, float]]:
        """
        Формирование эмоционального лексикона на основе WordNet
        
        Returns:
            Словарь вида {слово: {эмоция: вес, ...}, ...}
        """
        # Основные эмоции
        emotions = ["радость", "грусть", "злость", "страх", "удивление", "отвращение"]
        emotion_lexicon = {}
        
        # Получение списка всех слов в базе (или можно ограничиться частотным словарем)
        query = """
        MATCH (w:WordNetWord)
        RETURN w.name AS word
        LIMIT 5000
        """
        result = self.run_query(query)
        
        words = [item["word"] for item in result]
        
        # Для каждого слова ищем связь с эмоциями
        for word in words:
            word_emotions = {}
            
            for emotion in emotions:
                # Поиск кратчайшего пути от слова к эмоции
                query = f"""
                MATCH (w:WordNetWord {{name: $word}})
                MATCH (s:WordNetSynset)-[:IN_SYNSET]-(w)
                MATCH (e:WordNetWord {{name: $emotion}})
                MATCH (es:WordNetSynset)-[:IN_SYNSET]-(e)
                MATCH path = shortestPath((s)-[*1..4]-(es))
                RETURN length(path) AS distance
                ORDER BY distance ASC
                LIMIT 1
                """
                
                result = self.run_query(query, word=word, emotion=emotion)
                
                if result:
                    distance = result[0]["distance"]
                    # Вычисление веса на основе расстояния (чем меньше расстояние, тем больше вес)
                    weight = 1.0 / (1.0 + distance)
                    word_emotions[emotion] = weight
            
            # Если слово имеет эмоциональную окраску, добавляем в лексикон
            if word_emotions:
                emotion_lexicon[word] = word_emotions
        
        return emotion_lexicon
    
    async def export_emotion_lexicon(self, output_file: str = None):
        """
        Экспорт эмоционального лексикона в файл
        
        Args:
            output_file: Путь к выходному файлу
        """
        if output_file is None:
            output_file = str(WORDNET_DIR / "emotion_lexicon.json")
        
        # Получение лексикона
        lexicon = self.get_emotion_lexicon()
        
        # Сохранение в файл
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(lexicon, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Эмоциональный лексикон экспортирован в {output_file}")
        
        # Публикация события
        if self.nats_client and self.nats_client.connected:
            event = {
                "type": "emotion_lexicon_exported",
                "output_file": output_file,
                "entries_count": len(lexicon),
                "timestamp": time.time()
            }
            
            await self.nats_client.publish(
                "sasok.knowledge.wordnet.lexicon",
                json.dumps(event).encode()
            )

# Пример использования
async def main():
    # Получение настроек из переменных окружения
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    
    # Создание процессора WordNet
    processor = WordNetProcessor(neo4j_uri, neo4j_user, neo4j_password)
    await processor.initialize()
    
    # Импорт эмоциональной сети
    await processor.import_emotion_network(max_depth=2)
    
    # Экспорт эмоционального лексикона
    await processor.export_emotion_lexicon()
    
    # Закрытие соединений
    processor.close()

if __name__ == "__main__":
    asyncio.run(main())
