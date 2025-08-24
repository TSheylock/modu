"""
Система кэширования для SASOK

Обеспечивает эффективное кэширование результатов анализа эмоций
и других вычислительно затратных операций
"""
import time
import hashlib
import json
import asyncio
import logging
from typing import Dict, Any, Optional, Callable, TypeVar, Awaitable
from functools import wraps
import pickle
import os
from pathlib import Path

# Типы для аннотаций
T = TypeVar('T')
CacheKeyType = str
CacheValueType = Dict[str, Any]

# Настройка логгера
logger = logging.getLogger("CacheManager")

class CacheManager:
    """
    Менеджер кэша для оптимизации производительности SASOK
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """
        Синглтон для получения единого экземпляра кэша
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        # Кэш в памяти {key: {"value": any, "expiry": timestamp}}
        self.memory_cache: Dict[CacheKeyType, CacheValueType] = {}
        
        # Путь для персистентного кэша
        self.cache_dir = Path(os.environ.get("SASOK_CACHE_DIR", "/tmp/sasok_cache"))
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Ограничение размера кэша
        self.max_memory_items = 1000
        self.cleanup_frequency = 100  # Частота проверки и очистки кэша
        self.access_count = 0
        
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Генерирует уникальный ключ на основе аргументов
        """
        # Создаем композитный ключ из всех аргументов
        key_parts = [prefix]
        
        # Добавляем позиционные аргументы
        for arg in args:
            if isinstance(arg, (dict, list, tuple, set)):
                # Для сложных типов используем JSON сериализацию
                key_parts.append(json.dumps(arg, sort_keys=True))
            else:
                key_parts.append(str(arg))
                
        # Добавляем именованные аргументы в отсортированном порядке
        for k, v in sorted(kwargs.items()):
            key_parts.append(k)
            if isinstance(v, (dict, list, tuple, set)):
                key_parts.append(json.dumps(v, sort_keys=True))
            else:
                key_parts.append(str(v))
                
        # Генерируем MD5 хэш от всех частей ключа
        composite_key = "_".join(key_parts)
        return prefix + ":" + hashlib.md5(composite_key.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Получает значение из кэша
        
        Returns:
            Значение или None если ключа нет или он устарел
        """
        self.access_count += 1
        
        # Проверка на необходимость очистки кэша
        if self.access_count % self.cleanup_frequency == 0:
            self._cleanup_expired()
        
        # Проверяем кэш в памяти
        if key in self.memory_cache:
            cache_item = self.memory_cache[key]
            # Проверяем срок действия
            if "expiry" not in cache_item or cache_item["expiry"] > time.time():
                logger.debug(f"Кэш-хит (память): {key}")
                return cache_item["value"]
            else:
                # Удаляем просроченный элемент
                del self.memory_cache[key]
        
        # Проверяем персистентный кэш
        cache_file = self.cache_dir / f"{key}.cache"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cache_item = pickle.load(f)
                
                # Проверяем срок действия
                if "expiry" not in cache_item or cache_item["expiry"] > time.time():
                    # Копируем в память для быстрого доступа в будущем
                    self.memory_cache[key] = cache_item
                    logger.debug(f"Кэш-хит (диск): {key}")
                    return cache_item["value"]
                else:
                    # Удаляем просроченный файл
                    cache_file.unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"Ошибка при чтении кэша с диска: {e}")
        
        # Кэш-промах
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600, persist: bool = False) -> None:
        """
        Устанавливает значение в кэш
        
        Args:
            key: Ключ
            value: Значение
            ttl: Время жизни в секундах (по умолчанию 1 час)
            persist: Сохранять ли на диск
        """
        # Проверяем размер кэша
        if len(self.memory_cache) >= self.max_memory_items:
            self._evict_items()
            
        # Создаем элемент кэша
        expiry = time.time() + ttl if ttl > 0 else None
        cache_item = {"value": value, "expiry": expiry, "created": time.time()}
        
        # Сохраняем в памяти
        self.memory_cache[key] = cache_item
        
        # Если нужно, сохраняем на диск
        if persist:
            try:
                cache_file = self.cache_dir / f"{key}.cache"
                with open(cache_file, "wb") as f:
                    pickle.dump(cache_item, f)
                logger.debug(f"Кэш сохранен на диск: {key}")
            except Exception as e:
                logger.error(f"Ошибка при сохранении кэша на диск: {e}")
    
    def delete(self, key: str) -> bool:
        """
        Удаляет ключ из кэша
        
        Returns:
            bool: True если ключ был удален, False если ключа не было
        """
        was_in_memory = key in self.memory_cache
        if was_in_memory:
            del self.memory_cache[key]
        
        # Проверяем файл на диске
        cache_file = self.cache_dir / f"{key}.cache"
        was_on_disk = cache_file.exists()
        if was_on_disk:
            try:
                cache_file.unlink()
            except Exception as e:
                logger.error(f"Ошибка при удалении кэша с диска: {e}")
        
        return was_in_memory or was_on_disk
    
    def clear(self) -> None:
        """
        Очищает весь кэш
        """
        # Очистка памяти
        self.memory_cache.clear()
        
        # Очистка диска
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
        except Exception as e:
            logger.error(f"Ошибка при очистке кэша с диска: {e}")
    
    def _cleanup_expired(self) -> None:
        """
        Очищает просроченные элементы кэша
        """
        now = time.time()
        
        # Очистка памяти
        expired_keys = [
            key for key, item in self.memory_cache.items() 
            if "expiry" in item and item["expiry"] <= now
        ]
        
        for key in expired_keys:
            del self.memory_cache[key]
            
        if expired_keys:
            logger.debug(f"Очищено {len(expired_keys)} просроченных элементов из памяти")
        
        # Очистка диска делается периодически, не при каждом вызове
        if self.access_count % (self.cleanup_frequency * 10) == 0:
            try:
                expired_count = 0
                for cache_file in self.cache_dir.glob("*.cache"):
                    try:
                        with open(cache_file, "rb") as f:
                            cache_item = pickle.load(f)
                        
                        if "expiry" in cache_item and cache_item["expiry"] <= now:
                            cache_file.unlink()
                            expired_count += 1
                    except Exception:
                        # Если файл поврежден, удаляем его
                        cache_file.unlink(missing_ok=True)
                        expired_count += 1
                
                if expired_count:
                    logger.debug(f"Очищено {expired_count} просроченных элементов с диска")
            except Exception as e:
                logger.error(f"Ошибка при очистке кэша с диска: {e}")
    
    def _evict_items(self) -> None:
        """
        Вытесняет старые элементы из кэша в памяти
        """
        # Удаляем 25% самых старых элементов
        items_to_remove = max(1, len(self.memory_cache) // 4)
        
        # Сортируем по времени создания
        sorted_items = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1].get("created", 0)
        )
        
        # Удаляем старые элементы
        for key, _ in sorted_items[:items_to_remove]:
            del self.memory_cache[key]
            
        logger.debug(f"Вытеснено {items_to_remove} элементов из памяти")

# Декоратор для кэширования функций
def cached(ttl: int = 3600, persist: bool = False, prefix: str = "func"):
    """
    Декоратор для кэширования результатов функций
    
    Args:
        ttl: Время жизни кэша в секундах
        persist: Сохранять ли результат на диск
        prefix: Префикс для ключа кэша
    """
    cache_manager = CacheManager.get_instance()
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Генерируем ключ на основе аргументов
            cache_key = cache_manager._generate_key(f"{prefix}:{func.__name__}", *args, **kwargs)
            
            # Проверяем кэш
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Вызываем оригинальную функцию
            result = await func(*args, **kwargs)
            
            # Сохраняем результат в кэш
            cache_manager.set(cache_key, result, ttl, persist)
            
            return result
        return wrapper
    return decorator
