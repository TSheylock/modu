"""
Базовый класс для всех модулей SASOK.
Определяет общий интерфейс и функциональность модулей.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseModule(ABC):
    """Абстрактный базовый класс для всех модулей SASOK."""
    
    def __init__(self, event_bus, config):
        """
        Инициализация базового модуля.
        
        Args:
            event_bus: Шина событий для коммуникации между модулями
            config: Конфигурация системы
        """
        self.event_bus = event_bus
        self.config = config
        self.logger = self._setup_logger()
        self.active = False
        self.subscriptions = []
        self.state = {}
    
    def _setup_logger(self) -> logging.Logger:
        """Настройка логирования для модуля."""
        logger = logging.getLogger(f"SASOK_{self.__class__.__name__}")
        logger.setLevel(logging.INFO)  # Уровень можно настроить из конфигурации
        
        # Консольный обработчик
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '\033[1;35m%(asctime)s\033[0m | \033[1;34m%(name)s\033[0m | \033[1;33m%(levelname)s\033[0m | \033[1;32m%(message)s\033[0m'
        ))
        logger.addHandler(console_handler)
        
        return logger
    
    @abstractmethod
    async def initialize(self):
        """
        Инициализация модуля.
        Должна быть реализована в каждом конкретном модуле.
        """
        pass
    
    @abstractmethod
    async def activate(self):
        """
        Активация модуля.
        Должна быть реализована в каждом конкретном модуле.
        """
        pass
    
    @abstractmethod
    async def deactivate(self):
        """
        Деактивация модуля.
        Должна быть реализована в каждом конкретном модуле.
        """
        pass
    
    @abstractmethod
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обработка входящих данных.
        Должна быть реализована в каждом конкретном модуле.
        
        Args:
            data: Данные для обработки
            
        Returns:
            Результат обработки
        """
        pass
    
    async def subscribe(self, subject: str, callback):
        """
        Подписка на событие.
        
        Args:
            subject: Тема события
            callback: Функция обратного вызова
        """
        if self.event_bus:
            sub = await self.event_bus.subscribe(subject, cb=callback)
            self.subscriptions.append(sub)
            self.logger.info(f"Подписка на {subject} создана")
            return sub
        else:
            self.logger.error("Шина событий не инициализирована")
            return None
    
    async def publish(self, subject: str, data):
        """
        Публикация события.
        
        Args:
            subject: Тема события
            data: Данные события
        """
        if self.event_bus:
            await self.event_bus.publish(subject, data)
            self.logger.debug(f"Событие {subject} опубликовано")
        else:
            self.logger.error("Шина событий не инициализирована")
    
    async def get_state(self) -> Dict[str, Any]:
        """
        Получение текущего состояния модуля.
        
        Returns:
            Состояние модуля
        """
        return self.state
    
    async def set_state(self, state: Dict[str, Any]):
        """
        Установка состояния модуля.
        
        Args:
            state: Новое состояние
        """
        self.state = state
        await self.publish(f"{self.__class__.__name__.lower()}.state_changed", self.state)
    
    async def update_state(self, updates: Dict[str, Any]):
        """
        Обновление части состояния модуля.
        
        Args:
            updates: Обновления для состояния
        """
        self.state.update(updates)
        await self.publish(f"{self.__class__.__name__.lower()}.state_changed", self.state)