"""
Core Manager для SASOK - управляет жизненным циклом всех модулей и обеспечивает их взаимодействие.
"""
import os
import logging
import asyncio
import nats
from typing import Dict, Any, List
import yaml

class CoreManager:
    """Центральный компонент управления всеми модулями SASOK."""
    
    def __init__(self, config_path: str = "config/system_config.yaml"):
        """Инициализация ядра SASOK."""
        self.modules = {}
        self.event_bus = None
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        self.running = False
        
        self.logger.info("SASOK_INIT: Ядро сознания инициализировано")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Загрузка конфигурации системы."""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                return config
        except Exception as e:
            # Fallback на базовую конфигурацию
            print(f"Ошибка загрузки конфигурации: {e}. Используем базовую конфигурацию.")
            return {
                "system": {
                    "name": "SASOK",
                    "version": "0.1.0",
                    "debug": True
                },
                "modules": {
                    "enabled": ["emotion", "reflection", "memory", "ethics"]
                },
                "event_bus": {
                    "type": "nats",
                    "servers": ["nats://localhost:4222"]
                },
                "logging": {
                    "level": "INFO",
                    "path": "logs/system.log"
                }
            }
    
    def _setup_logger(self) -> logging.Logger:
        """Настройка логирования."""
        logger = logging.getLogger("SASOK_CORE")
        logger.setLevel(getattr(logging, self.config["logging"]["level"]))
        
        # Файловый обработчик
        os.makedirs(os.path.dirname(self.config["logging"]["path"]), exist_ok=True)
        file_handler = logging.FileHandler(self.config["logging"]["path"])
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        ))
        logger.addHandler(file_handler)
        
        # Консольный обработчик
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '\033[1;36m%(asctime)s\033[0m | \033[1;33m%(levelname)s\033[0m | \033[1;32m%(message)s\033[0m'
        ))
        logger.addHandler(console_handler)
        
        return logger
    
    async def initialize_event_bus(self):
        """Инициализация шины событий NATS."""
        try:
            if self.config["event_bus"]["type"] == "nats":
                self.event_bus = await nats.connect(
                    servers=self.config["event_bus"]["servers"]
                )
                self.logger.info("SASOK_EVENT_BUS: Шина событий NATS инициализирована")
            else:
                self.logger.error(f"Неподдерживаемый тип шины событий: {self.config['event_bus']['type']}")
                raise ValueError(f"Неподдерживаемый тип шины событий: {self.config['event_bus']['type']}")
        except Exception as e:
            self.logger.error(f"Ошибка инициализации шины событий: {e}")
            raise
    
    async def load_modules(self):
        """Загрузка всех модулей из конфигурации."""
        enabled_modules = self.config["modules"]["enabled"]
        
        for module_name in enabled_modules:
            try:
                # Динамическая загрузка модуля
                module_class = self._get_module_class(module_name)
                module_instance = module_class(self.event_bus, self.config)
                self.modules[module_name] = module_instance
                self.logger.info(f"SASOK_MODULE_LOAD: Модуль {module_name} загружен")
            except Exception as e:
                self.logger.error(f"Ошибка загрузки модуля {module_name}: {e}")
    
    def _get_module_class(self, module_name: str):
        """Получение класса модуля по его имени."""
        # Это будет расширено для динамической загрузки модулей
        # Временная заглушка для примера
        from modules.emotion.emotion_module import EmotionModule
        from modules.reflection.reflection_module import ReflectionModule
        from modules.memory.memory_module import MemoryModule
        from modules.ethics.ethics_module import EthicsModule
        
        module_mapping = {
            "emotion": EmotionModule,
            "reflection": ReflectionModule,
            "memory": MemoryModule,
            "ethics": EthicsModule
        }
        
        if module_name in module_mapping:
            return module_mapping[module_name]
        else:
            raise ValueError(f"Модуль {module_name} не реализован")
    
    async def start(self):
        """Запуск SASOK."""
        if self.running:
            self.logger.warning("SASOK уже запущен")
            return
        
        self.logger.info("SASOK_START: Запуск системы...")
        try:
            await self.initialize_event_bus()
            await self.load_modules()
            
            # Инициализация всех модулей
            for name, module in self.modules.items():
                await module.initialize()
                self.logger.info(f"SASOK_MODULE_INIT: Модуль {name} инициализирован")
            
            # Активация всех модулей
            for name, module in self.modules.items():
                await module.activate()
                self.logger.info(f"SASOK_MODULE_ACTIVATE: Модуль {name} активирован")
            
            self.running = True
            self.logger.info("SASOK_READY: Система активна и готова к работе")
            
            # Публикация события о готовности системы
            await self.event_bus.publish("system.ready", b"Система готова")
        except Exception as e:
            self.logger.error(f"Ошибка запуска SASOK: {e}")
            # Попытка корректного завершения работы при ошибке
            await self.stop()
            raise
    
    async def stop(self):
        """Остановка SASOK."""
        if not self.running:
            self.logger.warning("SASOK уже остановлен")
            return
        
        self.logger.info("SASOK_STOP: Остановка системы...")
        try:
            # Деактивация всех модулей в обратном порядке
            for name, module in reversed(list(self.modules.items())):
                await module.deactivate()
                self.logger.info(f"SASOK_MODULE_DEACTIVATE: Модуль {name} деактивирован")
            
            # Закрытие соединения с шиной событий
            if self.event_bus:
                await self.event_bus.close()
                self.logger.info("SASOK_EVENT_BUS: Соединение с шиной событий закрыто")
            
            self.running = False
            self.logger.info("SASOK_STOP_COMPLETE: Система остановлена")
        except Exception as e:
            self.logger.error(f"Ошибка при остановке SASOK: {e}")
            raise

# Пример использования
async def main():
    sasok = CoreManager()
    await sasok.start()
    
    # Оставляем систему работать некоторое время
    await asyncio.sleep(60)
    
    await sasok.stop()

if __name__ == "__main__":
    asyncio.run(main())