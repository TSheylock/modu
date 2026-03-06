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
    
    async def initialize(self):
        """Инициализация всех компонентов ядра (вызывается перед start)."""
        self.logger.info("SASOK_INIT: Подготовка компонентов ядра...")
        # Предзагрузка модулей без активации
        await self.load_modules()
        self.logger.info("SASOK_INIT: Компоненты ядра подготовлены")

    async def load_modules(self):
        """Загрузка всех модулей из конфигурации."""
        enabled_modules = self.config.get("modules", {}).get("enabled",
            ["emotion", "reflection", "memory", "ethics"])

        for module_name in enabled_modules:
            try:
                # Динамическая загрузка модуля
                module_class = self._get_module_class(module_name)
                if module_class:
                    module_instance = module_class(self.event_bus, self.config)
                    self.modules[module_name] = module_instance
                    self.logger.info(f"SASOK_MODULE_LOAD: Модуль {module_name} загружен")
            except Exception as e:
                self.logger.warning(f"Модуль {module_name} не загружен (degraded): {e}")

    def _get_module_class(self, module_name: str):
        """Получение класса модуля по его имени."""
        module_mapping = {}

        try:
            from modules.reflection.reflection_module import ReflectionModule
            module_mapping["reflection"] = ReflectionModule
        except ImportError as e:
            self.logger.warning(f"reflection: {e}")

        try:
            from modules.memory.memory_module import MemoryModule
            module_mapping["memory"] = MemoryModule
        except ImportError as e:
            self.logger.warning(f"memory: {e}")

        try:
            from modules.ethics.ethics_module import EthicsModule
            module_mapping["ethics"] = EthicsModule
        except ImportError as e:
            self.logger.warning(f"ethics: {e}")

        try:
            from modules.emotion.emotion_module import EmotionModule
            module_mapping["emotion"] = EmotionModule
        except ImportError:
            # Fallback: создадим обёрточный модуль из emotion_analysis
            try:
                from modules.emotion.emotion_adapter import EmotionModuleAdapter
                module_mapping["emotion"] = EmotionModuleAdapter
            except ImportError as e:
                self.logger.warning(f"emotion: {e}")

        try:
            from modules.dream.dream_adapter import DreamModule
            module_mapping["dream"] = DreamModule
        except ImportError:
            try:
                from modules.dream.dream_module import DreamModule
                module_mapping["dream"] = DreamModule
            except ImportError as e:
                self.logger.warning(f"dream: {e}")

        try:
            from modules.motivation.motivation_module import MotivationModule
            module_mapping["motivation"] = MotivationModule
        except ImportError as e:
            self.logger.warning(f"motivation: {e}")

        try:
            from modules.sasok_chain.sasok_chain_module import SASOKChainModule
            module_mapping["sasok_chain"] = SASOKChainModule
        except ImportError as e:
            self.logger.warning(f"sasok_chain: {e}")

        try:
            from modules.cognitive_migration.migration_module import CognitiveMigrationModule
            module_mapping["cognitive_migration"] = CognitiveMigrationModule
        except ImportError as e:
            self.logger.warning(f"cognitive_migration: {e}")

        try:
            from modules.zk_identity.zk_identity_module import ZKIdentityModule
            module_mapping["zk_identity"] = ZKIdentityModule
        except ImportError as e:
            self.logger.warning(f"zk_identity: {e}")

        try:
            from modules.snn_decoder.snn_module import SNNDecoderModule
            module_mapping["snn_decoder"] = SNNDecoderModule
        except ImportError as e:
            self.logger.warning(f"snn_decoder: {e}")

        try:
            from modules.symbiotic_engine.symbiotic_module import SymbioticEngineModule
            module_mapping["symbiotic_engine"] = SymbioticEngineModule
        except ImportError as e:
            self.logger.warning(f"symbiotic_engine: {e}")

        try:
            from modules.emotional_id.emotional_id_module import EmotionalIDModule
            module_mapping["emotional_id"] = EmotionalIDModule
        except ImportError as e:
            self.logger.warning(f"emotional_id: {e}")

        if module_name in module_mapping:
            return module_mapping[module_name]
        else:
            self.logger.warning(f"Модуль {module_name} не реализован")
            return None
    
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
            await self.event_bus.publish("system.ready", "Система готова".encode("utf-8"))
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