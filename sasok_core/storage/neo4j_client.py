"""
Neo4j клиент для SASOK

Обеспечивает интеграцию с графовой базой данных Neo4j
для хранения эмоциональных состояний, связей и метаданных
"""
import os
import json
import time
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from neo4j import GraphDatabase, basic_auth
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Настройка логгера
logger = logging.getLogger("Neo4jClient")

class Neo4jClient:
    """
    Клиент для взаимодействия с графовой базой данных Neo4j
    Используется для хранения эмоциональных данных в SASOK
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """
        Синглтон для получения единого экземпляра клиента
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def __init__(self):
        """
        Инициализация Neo4j клиента
        Получает параметры подключения из переменных окружения
        """
        # Получение настроек из переменных окружения
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "sasok")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        
        # Индикатор соединения
        self.connected = False
        self.driver = None
        
        # Подключение при инициализации
        self.connect()
        
    def connect(self) -> bool:
        """
        Установка соединения с Neo4j
        
        Returns:
            bool: Успешность подключения
        """
        try:
            # Создание драйвера
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=basic_auth(self.user, self.password)
            )
            
            # Проверка соединения
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS test").single()
                if result and result["test"] == 1:
                    self.connected = True
                    logger.info(f"Подключено к Neo4j: {self.uri}")
                    return True
                else:
                    logger.error("Не удалось проверить соединение с Neo4j")
                    self.connected = False
                    return False
                    
        except Exception as e:
            logger.error(f"Ошибка подключения к Neo4j: {e}")
            self.connected = False
            return False
            
    def close(self) -> None:
        """
        Закрытие соединения с Neo4j
        """
        if self.driver:
            self.driver.close()
            self.connected = False
            logger.info("Соединение с Neo4j закрыто")
            
    def _ensure_connected(self) -> bool:
        """
        Проверка соединения и переподключение при необходимости
        
        Returns:
            bool: Статус соединения
        """
        if not self.connected or not self.driver:
            return self.connect()
        return True
        
    def store_emotion(self, user_id: str, emotion_data: Dict[str, Any], 
                    metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Сохранение данных об эмоциях в Neo4j
        
        Args:
            user_id: Идентификатор пользователя
            emotion_data: Данные об эмоциях
            metadata: Дополнительные метаданные
            
        Returns:
            str: ID созданного узла или None в случае ошибки
        """
        if not self._ensure_connected():
            logger.error("Не удалось установить соединение с Neo4j")
            return None
            
        try:
            # Извлекаем основные данные
            emotion = emotion_data.get("dominant_emotion", "neutral")
            score = emotion_data.get("dominant_score", 0.5)
            timestamp = emotion_data.get("timestamp", time.time())
            
            # Конвертируем timestamp в ISO формат для Neo4j
            iso_time = datetime.fromtimestamp(timestamp).isoformat()
            
            # Извлекаем модальности
            modalities = []
            modality_data = {}
            
            for modality in ["text", "audio", "video"]:
                if modality in emotion_data:
                    modalities.append(modality)
                    modality_data[modality] = emotion_data[modality]
            
            # Подготовка метаданных
            if metadata is None:
                metadata = {}
                
            # Добавляем информацию о модальностях
            metadata["modalities"] = modalities
            
            # Конвертируем метаданные в JSON для хранения
            metadata_json = json.dumps(metadata)
            modality_data_json = json.dumps(modality_data)
            
            # Выполняем Cypher запрос для создания узлов и связей
            with self.driver.session(database=self.database) as session:
                result = session.execute_write(
                    self._create_emotion_node,
                    user_id=user_id,
                    emotion=emotion,
                    score=score,
                    timestamp=iso_time,
                    metadata=metadata_json,
                    modality_data=modality_data_json
                )
                
                if result and "node_id" in result:
                    logger.info(f"Эмоция сохранена в Neo4j: {emotion} для пользователя {user_id}")
                    return result["node_id"]
                else:
                    logger.error("Не удалось сохранить эмоциональные данные")
                    return None
                    
        except Exception as e:
            logger.error(f"Ошибка при сохранении эмоциональных данных: {e}")
            return None
            
    @staticmethod
    def _create_emotion_node(tx, user_id, emotion, score, timestamp, metadata, modality_data):
        """
        Транзакция для создания узла эмоции и связей в Neo4j
        
        Args:
            tx: Транзакция Neo4j
            user_id: Идентификатор пользователя
            emotion: Тип эмоции
            score: Интенсивность эмоции
            timestamp: Временная метка в ISO формате
            metadata: Метаданные в формате JSON
            modality_data: Данные о модальностях в формате JSON
            
        Returns:
            Dict: Результат операции
        """
        # Запрос для создания узлов и связей
        query = """
        // Находим или создаем узел пользователя
        MERGE (u:User {id: $user_id})
        
        // Создаем узел эмоции
        CREATE (e:Emotion {
            type: $emotion,
            score: $score,
            timestamp: datetime($timestamp),
            metadata: $metadata,
            modality_data: $modality_data,
            created_at: datetime()
        })
        
        // Создаем связь между пользователем и эмоцией
        CREATE (u)-[:EXPERIENCED {timestamp: datetime($timestamp)}]->(e)
        
        // Находим или создаем узел типа эмоции
        MERGE (et:EmotionType {name: $emotion})
        
        // Связываем эмоцию с типом
        CREATE (e)-[:IS_TYPE]->(et)
        
        // Возвращаем ID узла эмоции
        RETURN id(e) AS node_id
        """
        
        # Выполняем запрос
        result = tx.run(
            query,
            user_id=user_id,
            emotion=emotion,
            score=score,
            timestamp=timestamp,
            metadata=metadata,
            modality_data=modality_data
        )
        
        # Возвращаем первую запись
        record = result.single()
        if record:
            return {"node_id": str(record["node_id"])}
        return None
        
    def get_user_emotions(self, user_id: str, limit: int = 10, 
                         emotion_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Получение истории эмоций пользователя
        
        Args:
            user_id: Идентификатор пользователя
            limit: Максимальное количество записей
            emotion_type: Фильтр по типу эмоции
            
        Returns:
            List[Dict]: Список эмоциональных состояний
        """
        if not self._ensure_connected():
            logger.error("Не удалось установить соединение с Neo4j")
            return []
            
        try:
            with self.driver.session(database=self.database) as session:
                return session.execute_read(
                    self._get_user_emotions_tx,
                    user_id=user_id,
                    limit=limit,
                    emotion_type=emotion_type
                )
        except Exception as e:
            logger.error(f"Ошибка при получении эмоций пользователя: {e}")
            return []
            
    @staticmethod
    def _get_user_emotions_tx(tx, user_id, limit, emotion_type):
        """
        Транзакция для получения эмоций пользователя
        
        Args:
            tx: Транзакция Neo4j
            user_id: Идентификатор пользователя
            limit: Максимальное количество записей
            emotion_type: Фильтр по типу эмоции
            
        Returns:
            List[Dict]: Список эмоциональных состояний
        """
        # Строим запрос
        query = """
        MATCH (u:User {id: $user_id})-[:EXPERIENCED]->(e:Emotion)
        """
        
        # Добавляем фильтр по типу если указан
        if emotion_type:
            query += "WHERE e.type = $emotion_type "
            
        # Сортировка и лимит
        query += """
        RETURN e.type AS emotion, e.score AS score, 
               e.timestamp AS timestamp, e.metadata AS metadata,
               e.modality_data AS modality_data, id(e) AS node_id
        ORDER BY e.timestamp DESC
        LIMIT $limit
        """
        
        # Выполняем запрос
        result = tx.run(
            query,
            user_id=user_id,
            limit=limit,
            emotion_type=emotion_type
        )
        
        # Преобразуем результат
        emotions = []
        for record in result:
            # Преобразуем JSON строки обратно в объекты
            metadata = json.loads(record["metadata"]) if record["metadata"] else {}
            modality_data = json.loads(record["modality_data"]) if record["modality_data"] else {}
            
            # Создаем объект эмоции
            emotion = {
                "emotion": record["emotion"],
                "score": record["score"],
                "timestamp": record["timestamp"].isoformat(),
                "metadata": metadata,
                "modality_data": modality_data,
                "node_id": str(record["node_id"])
            }
            
            emotions.append(emotion)
            
        return emotions
        
    def get_emotion_graph(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """
        Получение графа эмоций пользователя за период
        
        Args:
            user_id: Идентификатор пользователя
            days: Количество дней для анализа
            
        Returns:
            Dict: Данные графа эмоций
        """
        if not self._ensure_connected():
            logger.error("Не удалось установить соединение с Neo4j")
            return {"nodes": [], "links": []}
            
        try:
            with self.driver.session(database=self.database) as session:
                return session.execute_read(
                    self._get_emotion_graph_tx,
                    user_id=user_id,
                    days=days
                )
        except Exception as e:
            logger.error(f"Ошибка при получении графа эмоций: {e}")
            return {"nodes": [], "links": []}
            
    @staticmethod
    def _get_emotion_graph_tx(tx, user_id, days):
        """
        Транзакция для получения графа эмоций
        
        Args:
            tx: Транзакция Neo4j
            user_id: Идентификатор пользователя
            days: Количество дней для анализа
            
        Returns:
            Dict: Данные графа эмоций
        """
        # Запрос для получения узлов эмоций
        nodes_query = """
        MATCH (u:User {id: $user_id})-[:EXPERIENCED]->(e:Emotion)
        WHERE datetime() - e.timestamp <= duration({days: $days})
        RETURN id(e) AS id, e.type AS type, e.score AS score, 
               e.timestamp AS timestamp
        ORDER BY e.timestamp
        """
        
        # Запрос для получения связей между последовательными эмоциями
        links_query = """
        MATCH (u:User {id: $user_id})-[:EXPERIENCED]->(e1:Emotion)
        WHERE datetime() - e1.timestamp <= duration({days: $days})
        WITH u, e1 ORDER BY e1.timestamp
        WITH u, collect(e1) AS emotions
        UNWIND range(0, size(emotions)-2) AS i
        WITH emotions[i] AS e1, emotions[i+1] AS e2
        RETURN id(e1) AS source, id(e2) AS target, 
               e1.type AS source_type, e2.type AS target_type,
               duration.between(e1.timestamp, e2.timestamp).seconds AS seconds_between
        """
        
        # Получаем узлы
        nodes_result = tx.run(nodes_query, user_id=user_id, days=days)
        nodes = []
        
        # Цвета для разных типов эмоций
        emotion_colors = {
            "joy": "#FFD700",
            "sadness": "#6495ED",
            "anger": "#DC143C",
            "fear": "#9932CC",
            "surprise": "#FF8C00",
            "disgust": "#2E8B57",
            "neutral": "#A9A9A9"
        }
        
        # Преобразуем узлы
        for record in nodes_result:
            nodes.append({
                "id": str(record["id"]),
                "type": record["type"],
                "score": record["score"],
                "timestamp": record["timestamp"].isoformat(),
                "color": emotion_colors.get(record["type"], "#CCCCCC"),
                "size": 10 + record["score"] * 20  # Размер узла зависит от интенсивности
            })
            
        # Получаем связи
        links_result = tx.run(links_query, user_id=user_id, days=days)
        links = []
        
        # Преобразуем связи
        for record in links_result:
            links.append({
                "source": str(record["source"]),
                "target": str(record["target"]),
                "value": 1,
                "seconds": record["seconds_between"],
                "label": f"{record['source_type']} → {record['target_type']}"
            })
            
        return {
            "nodes": nodes,
            "links": links
        }
        
    def get_emotional_insights(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Получение аналитических данных по эмоциям пользователя
        
        Args:
            user_id: Идентификатор пользователя
            days: Количество дней для анализа
            
        Returns:
            Dict: Аналитические данные
        """
        if not self._ensure_connected():
            logger.error("Не удалось установить соединение с Neo4j")
            return {}
            
        try:
            with self.driver.session(database=self.database) as session:
                return session.execute_read(
                    self._get_emotional_insights_tx,
                    user_id=user_id,
                    days=days
                )
        except Exception as e:
            logger.error(f"Ошибка при получении аналитики эмоций: {e}")
            return {}
            
    @staticmethod
    def _get_emotional_insights_tx(tx, user_id, days):
        """
        Транзакция для получения аналитики эмоций
        
        Args:
            tx: Транзакция Neo4j
            user_id: Идентификатор пользователя
            days: Количество дней для анализа
            
        Returns:
            Dict: Аналитические данные
        """
        # Запрос для подсчета эмоций по типам
        emotion_counts_query = """
        MATCH (u:User {id: $user_id})-[:EXPERIENCED]->(e:Emotion)
        WHERE datetime() - e.timestamp <= duration({days: $days})
        RETURN e.type AS emotion, count(e) AS count, avg(e.score) AS avg_score
        ORDER BY count DESC
        """
        
        # Запрос для анализа изменений эмоций
        emotion_changes_query = """
        MATCH (u:User {id: $user_id})-[:EXPERIENCED]->(e1:Emotion)
        WHERE datetime() - e1.timestamp <= duration({days: $days})
        WITH u, e1 ORDER BY e1.timestamp
        WITH u, collect(e1) AS emotions
        UNWIND range(0, size(emotions)-2) AS i
        WITH emotions[i] AS e1, emotions[i+1] AS e2
        WHERE e1.type <> e2.type
        RETURN e1.type AS from_emotion, e2.type AS to_emotion, count(*) AS changes
        ORDER BY changes DESC
        LIMIT 10
        """
        
        # Запрос для анализа тренда эмоций по дням
        emotion_trend_query = """
        MATCH (u:User {id: $user_id})-[:EXPERIENCED]->(e:Emotion)
        WHERE datetime() - e.timestamp <= duration({days: $days})
        RETURN date(e.timestamp) AS date, e.type AS emotion, 
               count(e) AS count, avg(e.score) AS avg_score
        ORDER BY date, count DESC
        """
        
        # Получаем данные по эмоциям
        emotion_counts = []
        for record in tx.run(emotion_counts_query, user_id=user_id, days=days):
            emotion_counts.append({
                "emotion": record["emotion"],
                "count": record["count"],
                "avg_score": record["avg_score"]
            })
            
        # Получаем данные по изменениям эмоций
        emotion_changes = []
        for record in tx.run(emotion_changes_query, user_id=user_id, days=days):
            emotion_changes.append({
                "from": record["from_emotion"],
                "to": record["to_emotion"],
                "count": record["changes"]
            })
            
        # Получаем данные по тренду эмоций
        emotion_trend = {}
        for record in tx.run(emotion_trend_query, user_id=user_id, days=days):
            date_str = record["date"].isoformat()
            emotion = record["emotion"]
            
            if date_str not in emotion_trend:
                emotion_trend[date_str] = {}
                
            emotion_trend[date_str][emotion] = {
                "count": record["count"],
                "avg_score": record["avg_score"]
            }
            
        # Формируем результат
        return {
            "emotion_counts": emotion_counts,
            "emotion_changes": emotion_changes,
            "emotion_trend": emotion_trend,
            "days_analyzed": days,
            "timestamp": datetime.now().isoformat()
        }
