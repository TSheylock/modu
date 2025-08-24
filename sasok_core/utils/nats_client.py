"""
Надежный NATS клиент для SASOK

Обеспечивает отказоустойчивое взаимодействие с NATS, включая:
- Автоматическое восстановление соединения
- Очередь сообщений при недоступности сервера
- Централизованное управление соединениями
"""
import asyncio
import json
import time
from typing import Dict, Any, Optional, List, Callable, Tuple
import nats
import logging
from functools import wraps

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NatsClient")

class NatsClient:
    _instance = None
    
    @classmethod
    async def get_instance(cls, nats_url="nats://localhost:4222"):
        """
        Синглтон для получения единого экземпляра NATS клиента
        """
        if cls._instance is None:
            cls._instance = cls(nats_url)
            await cls._instance.connect()
        return cls._instance
        
    def __init__(self, nats_url):
        """
        Инициализация NATS клиента
        
        Args:
            nats_url: URL для подключения к NATS серверу
        """
        self.nats_url = nats_url
        self.client = None
        self.connected = False
        self.message_queue = []
        self.subscriptions = {}
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 1.0  # seconds
        
    async def connect(self):
        """
        Установка соединения с NATS сервером
        """
        try:
            # Закрываем предыдущее соединение если оно существует
            if self.client and self.client.is_connected:
                await self.client.close()
            
            # Подключаемся к NATS
            self.client = await nats.connect(
                self.nats_url,
                reconnect_time_wait=self.reconnect_delay,
                max_reconnect_attempts=self.max_reconnect_attempts,
                error_cb=self._on_error,
                reconnected_cb=self._on_reconnect,
                disconnected_cb=self._on_disconnect,
                closed_cb=self._on_close
            )
            
            self.connected = True
            self.reconnect_attempts = 0
            logger.info(f"Подключено к NATS серверу по адресу {self.nats_url}")
            
            # Отправка сообщений из очереди
            await self._process_queued_messages()
            
            return True
        except Exception as e:
            self.connected = False
            self.reconnect_attempts += 1
            logger.error(f"Ошибка подключения к NATS: {e}")
            
            # Экспоненциальная задержка перед повторной попыткой
            delay = min(30, self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)))
            logger.info(f"Повторное подключение через {delay} сек (попытка {self.reconnect_attempts})")
            
            return False
            
    async def _process_queued_messages(self):
        """
        Обработка сообщений из очереди при восстановлении соединения
        """
        if not self.message_queue:
            return
            
        logger.info(f"Отправка {len(self.message_queue)} сообщений из очереди")
        
        sent_messages = []
        for subject, payload in self.message_queue:
            try:
                await self.client.publish(subject, payload)
                sent_messages.append((subject, payload))
                logger.debug(f"Отправлено отложенное сообщение: {subject}")
            except Exception as e:
                logger.error(f"Ошибка при отправке сообщения из очереди: {e}")
                break
                
        # Удаляем отправленные сообщения из очереди
        for msg in sent_messages:
            self.message_queue.remove(msg)
            
        if self.message_queue:
            logger.info(f"Осталось {len(self.message_queue)} сообщений в очереди")
                
    async def publish(self, subject: str, data: Dict[str, Any], headers: Dict[str, str] = None) -> bool:
        """
        Публикация сообщения в NATS
        
        Args:
            subject: Тема сообщения
            data: Данные для отправки (будут сконвертированы в JSON)
            headers: Заголовки сообщения (опционально)
            
        Returns:
            bool: Успешность публикации
        """
        payload = json.dumps(data).encode()
        
        # Если нет соединения, пробуем подключиться
        if not self.connected:
            success = await self.connect()
            if not success:
                # Добавляем сообщение в очередь для отправки позже
                self.message_queue.append((subject, payload))
                logger.warning(f"Сообщение добавлено в очередь: {subject}")
                return False
                
        try:
            # Отправка сообщения с заголовками если они есть
            if headers:
                await self.client.publish(subject, payload, headers=headers)
            else:
                await self.client.publish(subject, payload)
            return True
        except Exception as e:
            logger.error(f"Ошибка публикации в {subject}: {e}")
            # Добавляем сообщение в очередь
            self.message_queue.append((subject, payload))
            self.connected = False
            return False
               
    async def subscribe(self, subject: str, callback: Callable, queue: str = None) -> str:
        """
        Подписка на тему в NATS
        
        Args:
            subject: Тема для подписки
            callback: Функция обратного вызова
            queue: Имя очереди для балансировки нагрузки
            
        Returns:
            str: ID подписки или None в случае ошибки
        """
        if not self.connected:
            success = await self.connect()
            if not success:
                logger.error("Невозможно подписаться: нет соединения")
                return None
                
        try:
            # Оборачиваем callback для обработки ошибок
            async def safe_callback(msg):
                try:
                    await callback(msg)
                except Exception as e:
                    logger.error(f"Ошибка в обработчике подписки: {e}")
            
            # Создаем подписку
            sub = await self.client.subscribe(
                subject, 
                cb=safe_callback,
                queue=queue
            )
            
            # Сохраняем подписку
            self.subscriptions[sub.sid] = {
                "subject": subject,
                "queue": queue,
                "subscription": sub
            }
            
            logger.info(f"Успешная подписка на {subject} (sid: {sub.sid})")
            return sub.sid
        except Exception as e:
            logger.error(f"Ошибка подписки на {subject}: {e}")
            return None
    
    async def unsubscribe(self, sid: str) -> bool:
        """
        Отписка от темы
        
        Args:
            sid: ID подписки
            
        Returns:
            bool: Успешность отписки
        """
        if sid not in self.subscriptions:
            logger.warning(f"Подписка с ID {sid} не найдена")
            return False
            
        try:
            # Отписываемся
            await self.subscriptions[sid]["subscription"].unsubscribe()
            # Удаляем из словаря
            del self.subscriptions[sid]
            logger.info(f"Отписка от {sid} выполнена успешно")
            return True
        except Exception as e:
            logger.error(f"Ошибка отписки от {sid}: {e}")
            return False
            
    # Обработчики событий NATS
    async def _on_error(self, e):
        logger.error(f"NATS ошибка: {e}")
    
    async def _on_reconnect(self):
        logger.info("NATS соединение восстановлено")
        self.connected = True
        self.reconnect_attempts = 0
        await self._process_queued_messages()
    
    async def _on_disconnect(self):
        logger.warning("NATS соединение разорвано")
        self.connected = False
    
    async def _on_close(self):
        logger.info("NATS соединение закрыто")
        self.connected = False
               
    async def close(self):
        """
        Закрытие соединения с NATS
        """
        if self.client and self.client.is_connected:
            # Отписываемся от всех тем
            for sid in list(self.subscriptions.keys()):
                await self.unsubscribe(sid)
                
            # Закрываем соединение
            await self.client.close()
            self.connected = False
            logger.info("NATS соединение закрыто")


# Декоратор для отслеживания производительности
def track_performance(category: str):
    """
    Декоратор для отслеживания производительности функций
    
    Args:
        category: Категория для классификации метрик
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.debug(f"Performance[{category}]: {func.__name__} took {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Error[{category}]: {func.__name__} failed after {elapsed:.3f}s: {e}")
                raise
        return wrapper
    return decorator
