
## Модуль памяти (modules/memory/memory_module.py)

```python
"""
Модуль памяти SASOK.
Долговременное хранение ключевых событий, состояний, уроков и опыта.
"""
import json
import uuid
import datetime
import asyncio
import sqlite3
from typing import Dict, Any, List, Optional, Tuple
from core.base_module import BaseModule

# Для работы с базой данных SQLite
class EpisodicMemoryDB:
    """Управляет эпизодической памятью в SQLite."""
    
    def __init__(self, db_path: str = "modules/memory/episodic_mem.db"):
        """
        Инициализация базы данных эпизодической памяти.
        
        Args:
            db_path: Путь к файлу базы данных
        """
        self.db_path = db_path
        self.connection = None
        self.initialize_db()
    
    def initialize_db(self):
        """Инициализация схемы базы данных."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            cursor = self.connection.cursor()
            
            # Создание таблицы эпизодов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    importance REAL NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT
                )
            ''')
            
            # Создание таблицы тегов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tags (
                    episode_id TEXT,
                    tag TEXT,
                    PRIMARY KEY (episode_id, tag),
                    FOREIGN KEY (episode_id) REFERENCES episodes (id)
                )
            ''')
            
            # Создание таблицы связей между эпизодами
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS episode_relations (
                    source_id TEXT,
                    target_id TEXT,
                    relation_type TEXT NOT NULL,
                    strength REAL NOT NULL,
                    PRIMARY KEY (source_id, target_id, relation_type),
                    FOREIGN KEY (source_id) REFERENCES episodes (id),
                    FOREIGN KEY (target_id) REFERENCES episodes (id)
                )
            ''')
            
            # Создание индексов для оптимизации запросов
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_episodes_type ON episodes (type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_episodes_importance ON episodes (importance)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags (tag)')
            
            self.connection.commit()
        except Exception as e:
            print(f"Ошибка инициализации базы данных: {e}")
            if self.connection:
                self.connection.close()
            self.connection = None
    
    def add_episode(self, episode: Dict[str, Any]) -> str:
        """
        Добавление нового эпизода в память.
        
        Args:
            episode: Данные эпизода
            
        Returns:
            Идентификатор эпизода
        """
        if not self.connection:
            self.initialize_db()
            if not self.connection:
                raise Exception("Не удалось подключиться к базе данных")
        
        try:
            # Генерация идентификатора, если не указан
            episode_id = episode.get("id", str(uuid.uuid4()))
            episode["id"] = episode_id
            
            # Преобразование метаданных в JSON
            metadata = episode.get("metadata", {})
            metadata_json = json.dumps(metadata) if metadata else None
            
            # Добавление эпизода
            cursor = self.connection.cursor()
            cursor.execute(
                '''
                INSERT INTO episodes (id, timestamp, type, source, importance, content, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    episode_id,
                    episode.get("timestamp", datetime.datetime.now().isoformat()),
                    episode.get("type", "unknown"),
                    episode.get("source", "unknown"),
                    episode.get("importance", 0.5),
                    episode.get("content", ""),
                    metadata_json
                )
            )
            
            # Добавление тегов
            tags = episode.get("tags", [])
            for tag in tags:
                cursor.execute(
                    '''
                    INSERT INTO tags (episode_id, tag) VALUES (?, ?)
                    ''',
                    (episode_id, tag)
                )
            
            # Добавление связей
            relations = episode.get("relations", [])
            for relation in relations:
                if "target_id" in relation and "relation_type" in relation:
                    cursor.execute(
                        '''
                        INSERT INTO episode_relations (source_id, target_id, relation_type, strength)
                        VALUES (?, ?, ?, ?)
                        ''',
                        (
                            episode_id,
                            relation["target_id"],
                            relation["relation_type"],
                            relation.get("strength", 0.5)
                        )
                    )
            
            self.connection.commit()
            return episode_id
        except Exception as e:
            self.connection.rollback()
            raise Exception(f"Ошибка добавления эпизода: {e}")
    
    def get_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение эпизода по идентификатору.
        
        Args:
            episode_id: Идентификатор эпизода
            
        Returns:
            Эпизод или None, если не найден
        """
        if not self.connection:
            self.initialize_db()
            if not self.connection:
                raise Exception("Не удалось подключиться к базе данных")
        
        try:
            cursor = self.connection.cursor()
            
            # Получение эпизода
            cursor.execute(
                '''
                SELECT id, timestamp, type, source, importance, content, metadata
                FROM episodes
                WHERE id = ?
                ''',
                (episode_id,)
            )
            
            episode_data = cursor.fetchone()
            if not episode_data:
                return None
            
            # Преобразование в словарь
            episode = {
                "id": episode_data[0],
                "timestamp": episode_data[1],
                "type": episode_data[2],
                "source": episode_data[3],
                "importance": episode_data[4],
                "content": episode_data[5]
            }
            
            # Добавление метаданных, если есть
            if episode_data[6]:
                try:
                    episode["metadata"] = json.loads(episode_data[6])
                except:
                    episode["metadata"] = {}
            
            # Получение тегов
            cursor.execute(
                '''
                SELECT tag FROM tags WHERE episode_id = ?
                ''',
                (episode_id,)
            )
            
            tags = [row[0] for row in cursor.fetchall()]
            episode["tags"] = tags
            
            # Получение связей
            cursor.execute(
                '''
                SELECT target_id, relation_type, strength
                FROM episode_relations
                WHERE source_id = ?
                ''',
                (episode_id,)
            )
            
            relations = []
            for row in cursor.fetchall():
                relations.append({
                    "target_id": row[0],
                    "relation_type": row[1],
                    "strength": row[2]
                })
            
            episode["relations"] = relations
            
            return episode
        except Exception as e:
            raise Exception(f"Ошибка получения эпизода: {e}")
    
    def update_episode_importance(self, episode_id: str, importance: float) -> bool:
        """
        Обновление важности эпизода.
        
        Args:
            episode_id: Идентификатор эпизода
            importance: Новая важность
            
        Returns:
            True, если обновление успешно, иначе False
        """
        if not self.connection:
            self.initialize_db()
            if not self.connection:
                return False
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                '''
                UPDATE episodes SET importance = ? WHERE id = ?
                ''',
                (importance, episode_id)
            )
            
            self.connection.commit()
            return cursor.rowcount > 0
        except Exception as e:
            self.connection.rollback()
            print(f"Ошибка обновления важности эпизода: {e}")
            return False
    
    def add_relation(self, source_id: str, target_id: str, relation_type: str, strength: float) -> bool:
        """
        Добавление связи между эпизодами.
        
        Args:
            source_id: Идентификатор исходного эпизода
            target_id: Идентификатор целевого эпизода
            relation_type: Тип связи
            strength: Сила связи
            
        Returns:
            True, если добавление успешно, иначе False
        """
        if not self.connection:
            self.initialize_db()
            if not self.connection:
                return False
        
        try:
            cursor = self.connection.cursor()
            
            # Проверка существования эпизодов
            cursor.execute(
                '''
                SELECT COUNT(*) FROM episodes WHERE id IN (?, ?)
                ''',
                (source_id, target_id)
            )
            
            if cursor.fetchone()[0] < 2:
                return False
            
            # Добавление или обновление связи
            cursor.execute(
                '''
                INSERT OR REPLACE INTO episode_relations (source_id, target_id, relation_type, strength)
                VALUES (?, ?, ?, ?)
                ''',
                (source_id, target_id, relation_type, strength)
            )
            
            self.connection.commit()
            return True
        except Exception as e:
            self.connection.rollback()
            print(f"Ошибка добавления связи: {e}")
            return False
    
    def search_episodes(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Поиск эпизодов по критериям.
        
        Args:
            query: Критерии поиска
            limit: Максимальное количество результатов
            
        Returns:
            Список найденных эпизодов
        """
        if not self.connection:
            self.initialize_db()
            if not self.connection:
                return []
        
        try:
            sql_parts = ["SELECT id FROM episodes WHERE 1=1"]
            params = []
            
            # Фильтрация по типу
            if "type" in query:
                sql_parts.append("AND type = ?")
                params.append(query["type"])
            
            # Фильтрация по источнику
            if "source" in query:
                sql_parts.append("AND source = ?")
                params.append(query["source"])
            
            # Фильтрация по минимальной важности
            if "min_importance" in query:
                sql_parts.append("AND importance >= ?")
                params.append(query["min_importance"])
            
            # Фильтрация по временному диапазону
            if "start_time" in query:
                sql_parts.append("AND timestamp >= ?")
                params.append(query["start_time"])
            
            if "end_time" in query:
                sql_parts.append("AND timestamp <= ?")
                params.append(query["end_time"])
            
            # Фильтрация по содержанию
            if "content_like" in query:
                sql_parts.append("AND content LIKE ?")
                params.append(f"%{query['content_like']}%")
            
            # Фильтрация по тегам
            if "tags" in query and query["tags"]:
                placeholders = ", ".join(["?"] * len(query["tags"]))
                sql_parts.append(f"""
                    AND id IN (
                        SELECT episode_id FROM tags
                        WHERE tag IN ({placeholders})
                        GROUP BY episode_id
                        HAVING COUNT(DISTINCT tag) = {len(query["tags"])}
                    )
                """)
                params.extend(query["tags"])
            
            # Сортировка и лимит
            sql_parts.append("ORDER BY importance DESC, timestamp DESC")
            sql_parts.append("LIMIT ?")
            params.append(limit)
            
            # Выполнение запроса
            cursor = self.connection.cursor()
            cursor.execute(" ".join(sql_parts), params)
            
            # Получение найденных эпизодов
            episode_ids = [row[0] for row in cursor.fetchall()]
            return [self.get_episode(episode_id) for episode_id in episode_ids]
        except Exception as e:
            print(f"Ошибка поиска эпизодов: {e}")
            return []
    
    def get_related_episodes(self, episode_id: str, relation_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Получение связанных эпизодов.
        
        Args:
            episode_id: Идентификатор эпизода
            relation_type: Тип связи (опционально)
            limit: Максимальное количество результатов
            
        Returns:
            Список связанных эпизодов
        """
        if not self.connection:
            self.initialize_db()
            if not self.connection:
                return []
        
        try:
            cursor = self.connection.cursor()
            
            # Запрос связей
            if relation_type:
                cursor.execute(
                    '''
                    SELECT target_id FROM episode_relations
                    WHERE source_id = ? AND relation_type = ?
                    ORDER BY strength DESC
                    LIMIT ?
                    ''',
                    (episode_id, relation_type, limit)
                )
            else:
                cursor.execute(
                    '''
                    SELECT target_id FROM episode_relations
                    WHERE source_id = ?
                    ORDER BY strength DESC
                    LIMIT ?
                    ''',
                    (episode_id, limit)
                )
            
            # Получение связанных эпизодов
            target_ids = [row[0] for row in cursor.fetchall()]
            return [self.get_episode(target_id) for target_id in target_ids]
        except Exception as e:
            print(f"Ошибка получения связанных эпизодов: {e}")
            return []
    
    def close(self):
        """Закрытие соединения с базой данных."""
        if self.connection:
            self.connection.close()
            self.connection = None


class MemoryModule(BaseModule):
    """Модуль памяти SASOK."""
    
    async def initialize(self):
        """Инициализация модуля памяти."""
        self.logger.info("Инициализация модуля памяти...")
        
        # Создание компонентов памяти
        self.episodic_memory = EpisodicMemoryDB()
        self.state_snapshots = {}
        
        # Инициализация состояния
        self.state = {
            "active": False,
            "episode_count": 0,
            "last_episode": None,
            "memory_intensity": 0.5
        }
        
        # Загрузка количества эпизодов
        self._update_episode_count()
        
        self.logger.info("Модуль памяти инициализирован")
    
    async def activate(self):
        """Активация модуля памяти."""
        if self.active:
            self.logger.warning("Модуль памяти уже активен")
            return
        
        self.logger.info("Активация модуля памяти...")
        
        # Подписка на события для памяти
        await self.subscribe("emotion.state_changed", self._on_emotion_changed)
        await self.subscribe("interaction.completed", self._on_interaction_completed)
        await self.subscribe("reflection.insight_generated", self._on_reflection_insight)
        await self.subscribe("decision.made", self._on_decision_made)
        
        # Запуск периодической консолидации памяти
        asyncio.create_task(self._periodic_memory_consolidation())
        
        self.active = True
        await self.update_state({"active": True})
        
        self.logger.info("Модуль памяти активирован")
    
    async def deactivate(self):
        """Деактивация модуля памяти."""
        if not self.active:
            self.logger.warning("Модуль памяти уже неактивен")
            return
        
        self.logger.info("Деактивация модуля памяти...")
        
        # Отписка от всех событий
        for subscription in self.subscriptions:
            await subscription.unsubscribe()
        self.subscriptions = []
        
        # Закрытие соединения с базой данных
        self.episodic_memory.close()
        
        self.active = False
        await self.update_state({"active": False})
        
        self.logger.info("Модуль памяти деактивирован")
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обработка входящих данных для памяти.
        
        Args:
            data: Данные для обработки
            
        Returns:
            Результат обработки
        """
        if not self.active:
            self.logger.warning("Попытка обработки данных неактивным модулем памяти")
            return {"error": "Module inactive"}
        
        self.logger.info(f"Обработка данных для памяти: {data.get('type', 'unknown')}")
        
        try:
            result = {"processed": True}
            
            # Обработка в зависимости от типа данных
            if data.get("type") == "memory_store":
                result.update(await self._process_memory_store(data))
            elif data.get("type") == "memory_query":
                result.update(await self._process_memory_query(data))
            elif data.get("type") == "memory_relation":
                result.update(await self._process_memory_relation(data))
            elif data.get("type") == "state_snapshot":
                result.update(await self._process_state_snapshot(data))
            else:
                self.logger.warning(f"Неизвестный тип данных для памяти: {data.get('type')}")
                result["processed"] = False
                result["error"] = "Unknown data type"
            
            return result
        except Exception as e:
            self.logger.error(f"Ошибка при обработке данных для памяти: {e}")
            return {"error": str(e), "processed": False}
    
    async def _process_memory_store(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка запроса на сохранение в память."""
        episode_data = data.get("episode", {})
        
        # Проверка наличия необходимых полей
        if not episode_data.get("content"):
            return {"error": "Missing content in episode", "processed": False}
        
        # Дополнение данных эпизода
        if "timestamp" not in episode_data:
            episode_data["timestamp"] = datetime.datetime.now().isoformat()
        
        if "type" not in episode_data:
            episode_data["type"] = data.get("source", "unknown")
        
        # Добавление эпизода
        episode_id = self.episodic_memory.add_episode(episode_data)
        
        # Обновление состояния
        self._update_episode_count()
        self.state["last_episode"] = {
            "id": episode_id,
            "timestamp": episode_data["timestamp"],
            "type": episode_data["type"]
        }
        
        # Публикация события о новом эпизоде
        await self.publish(
            "memory.episode_stored", 
            json.dumps({
                "episode_id": episode_id,
                "type": episode_data["type"],
                "importance": episode_data.get("importance", 0.5),
                "tags": episode_data.get("tags", [])
            })
        )
        
        # Проверка на значимость эпизода
        if episode_data.get("importance", 0.5) > 0.7:
            await self.publish(
                "memory.significant_recall", 
                json.dumps({
                    "memory": {
                        "id": episode_id,
                        "content": episode_data.get("content", ""),
                        "importance": episode_data.get("importance", 0.5),
                        "tags": episode_data.get("tags", [])
                    }
                })
            )
        
        return {
            "episode_id": episode_id,
            "successful": True
        }
    
    async def _process_memory_query(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка запроса на поиск в памяти."""
        query_type = data.get("query_type", "search")
        
        if query_type == "get_episode":
            # Получение конкретного эпизода
            episode_id = data.get("episode_id")
            if not episode_id:
                return {"error": "Missing episode_id", "processed": False}
            
            episode = self.episodic_memory.get_episode(episode_id)
            return {"episode": episode}
        
        elif query_type == "search":
            # Поиск эпизодов по критериям
            query = data.get("query", {})
            limit = data.get("limit", 10)
            
            episodes = self.episodic_memory.search_episodes(query, limit)
            return {"episodes": episodes, "count": len(episodes)}
        
        elif query_type == "related":
            # Поиск связанных эпизодов
            episode_id = data.get("episode_id")
            relation_type = data.get("relation_type")
            limit = data.get("limit", 10)
            
            if not episode_id:
                return {"error": "Missing episode_id", "processed": False}
            
            episodes = self.episodic_memory.get_related_episodes(episode_id, relation_type, limit)
            return {"episodes": episodes, "count": len(episodes)}
        
        else:
            return {"error": f"Unknown query_type: {query_type}", "processed": False}
    
    async def _process_memory_relation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка запроса на создание связи между эпизодами."""
        source_id = data.get("source_id")
        target_id = data.get("target_id")
        relation_type = data.get("relation_type")
        strength = data.get("strength", 0.5)
        
        # Проверка наличия необходимых полей
        if not source_id or not target_id or not relation_type:
            return {"error": "Missing relation data", "processed": False}
        
        # Добавление связи
        success = self.episodic_memory.add_relation(source_id, target_id, relation_type, strength)
        
        return {
            "successful": success,
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": relation_type
        }
    
    async def _process_state_snapshot(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка снимка состояния."""
        snapshot_data = data.get("snapshot", {})
        source = data.get("source", "unknown")
        
        # Генерация идентификатора снимка
        snapshot_id = f"{source}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Сохранение снимка
        self.state_snapshots[snapshot_id] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "source": source,
            "data": snapshot_data
        }
        
        # Очистка старых снимков (оставляем только 100 последних)
        if len(self.state_snapshots) > 100:
            oldest_key = min(self.state_snapshots.keys(), key=lambda k: self.state_snapshots[k]["timestamp"])
            del self.state_snapshots[oldest_key]
        
        return {
            "snapshot_id": snapshot_id,
            "successful": True
        }
    
    def _update_episode_count(self):
        """Обновление количества эпизодов в состоянии."""
        try:
            # Простой запрос для подсчета эпизодов
            count = len(self.episodic_memory.search_episodes({}, 9999))
            self.state["episode_count"] = count
        except Exception as e:
            self.logger.error(f"Ошибка при подсчете эпизодов: {e}")
    
    async def _on_emotion_changed(self, msg):
        """Обработчик события изменения эмоционального состояния."""
        data = json.loads(msg.data.decode())
        emotions = data.get("emotions", {})
        
        # Сохранение снимка эмоционального состояния
        await self.process({
            "type": "state_snapshot",
            "source": "emotion_module",
            "snapshot": {
                "emotions": emotions,
                "context": data.get("context", "unknown")
            }
        })
        
        # Сохранение эпизода только для значимых эмоциональных состояний
        max_emotion = max(emotions.items(), key=lambda x: x[1]) if emotions else (None, 0)
        if max_emotion[1] > 0.6:
            await self.process({
                "type": "memory_store",
                "source": "emotion_module",
                "episode": {
                    "type": "emotion",
                    "content": f"Значимое эмоциональное состояние