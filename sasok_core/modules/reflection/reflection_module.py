"""
Модуль рефлексии SASOK.
Анализирует внутренние процессы, переоценка решений, настройка приоритетов.
"""
import json
import asyncio
import datetime
from typing import Dict, Any, List
from core.base_module import BaseModule

class ReflectionEngine:
    """Основной механизм рефлексии."""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.error_map = {}
        self.retrospection_cache = {}
        self.meta_intents = self._load_meta_intents()
    
    def _load_meta_intents(self) -> Dict[str, Any]:
        """Загрузка мета-интентов системы."""
        try:
            with open("modules/reflection/meta_intents.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Ошибка загрузки мета-интентов: {e}")
            # Базовые мета-интенты по умолчанию
            return {
                "self_improvement": {
                    "priority": 0.8,
                    "description": "Стремление к самосовершенствованию"
                },
                "user_understanding": {
                    "priority": 0.9,
                    "description": "Стремление к пониманию пользователя"
                },
                "emotional_stability": {
                    "priority": 0.7,
                    "description": "Поддержание эмоциональной стабильности"
                }
            }
    
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Анализ входящих данных для рефлексии.
        
        Args:
            data: Данные для анализа
            
        Returns:
            Результат рефлексии
        """
        analysis = {
            "timestamp": datetime.datetime.now().isoformat(),
            "source": data.get("source", "unknown"),
            "insights": [],
            "recommended_actions": []
        }
        
        # Определение типа данных для соответствующего анализа
        if "emotion" in data:
            analysis.update(await self._analyze_emotion(data))
        elif "decision" in data:
            analysis.update(await self._analyze_decision(data))
        elif "error" in data:
            analysis.update(await self._analyze_error(data))
        elif "interaction" in data:
            analysis.update(await self._analyze_interaction(data))
        
        # Сохранение в кэш ретроспекции
        cache_key = f"{data.get('source', 'unknown')}_{datetime.datetime.now().isoformat()}"
        self.retrospection_cache[cache_key] = analysis
        
        return analysis
    
    async def _analyze_emotion(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ эмоциональных данных."""
        emotion_data = data.get("emotion", {})
        emotion_type = emotion_data.get("type", "unknown")
        emotion_intensity = emotion_data.get("intensity", 0.5)
        
        insights = []
        actions = []
        
        # Анализ на основе типа эмоции
        if emotion_type in ["anger", "frustration"]:
            insights.append("Повышенная эмоциональная напряженность может требовать регуляции")
            actions.append({
                "target": "emotion_module",
                "action": "regulate",
                "parameters": {"intensity_target": min(emotion_intensity, 0.6)}
            })
        elif emotion_type in ["joy", "satisfaction"]:
            insights.append("Положительные эмоции способствуют когнитивной гибкости")
            actions.append({
                "target": "motivation_module",
                "action": "amplify",
                "parameters": {"goal": "user_engagement"}
            })
        
        # Сравнение с историческими данными эмоций (if available)
        # ...
        
        return {
            "insights": insights,
            "recommended_actions": actions
        }
    
    async def _analyze_decision(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ процесса принятия решений."""
        decision_data = data.get("decision", {})
        decision_outcome = decision_data.get("outcome", "unknown")
        decision_confidence = decision_data.get("confidence", 0.5)
        
        insights = []
        actions = []
        
        # Анализ на основе уверенности в решении
        if decision_confidence < 0.4:
            insights.append("Низкая уверенность в решении указывает на необходимость улучшения процесса")
            if decision_outcome == "negative":
                # Запись в карту ошибок
                error_key = f"{decision_data.get('context', 'unknown')}_{decision_data.get('type', 'unknown')}"
                self.error_map[error_key] = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "description": decision_data.get("description", ""),
                    "confidence": decision_confidence
                }
                
                actions.append({
                    "target": "self_regulation_module",
                    "action": "adjust_decision_process",
                    "parameters": {"context": decision_data.get("context")}
                })
        
        return {
            "insights": insights,
            "recommended_actions": actions
        }
    
    async def _analyze_error(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ ошибок."""
        error_data = data.get("error", {})
        error_type = error_data.get("type", "unknown")
        error_severity = error_data.get("severity", "medium")
        
        insights = []
        actions = []
        
        # Регистрация ошибки в карте ошибок
        error_key = f"{error_type}_{datetime.datetime.now().isoformat()}"
        self.error_map[error_key] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "description": error_data.get("description", ""),
            "severity": error_severity,
            "context": error_data.get("context", {})
        }
        
        # Анализ по типу ошибки
        if error_type == "processing_error":
            insights.append("Ошибка обработки может указывать на неправильные входные данные или логику")
            actions.append({
                "target": "core_manager",
                "action": "review_module_logic",
                "parameters": {"module": error_data.get("module", "unknown")}
            })
        elif error_type == "communication_error":
            insights.append("Ошибка коммуникации указывает на проблемы в шине событий")
            actions.append({
                "target": "core_manager",
                "action": "check_event_bus",
                "parameters": {}
            })
        
        return {
            "insights": insights,
            "recommended_actions": actions
        }
    
    async def _analyze_interaction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ взаимодействия с пользователем."""
        interaction_data = data.get("interaction", {})
        interaction_type = interaction_data.get("type", "unknown")
        interaction_quality = interaction_data.get("quality", 0.5)
        
        insights = []
        actions = []
        
        # Анализ на основе качества взаимодействия
        if interaction_quality < 0.4:
            insights.append("Низкое качество взаимодействия требует улучшения коммуникации")
            actions.append({
                "target": "interaction_module",
                "action": "improve_engagement",
                "parameters": {"context": interaction_data.get("context")}
            })
        
        return {
            "insights": insights,
            "recommended_actions": actions
        }
    
    def get_meta_intent(self, intent_name: str) -> Dict[str, Any]:
        """
        Получение мета-интента по имени.
        
        Args:
            intent_name: Имя мета-интента
        
        Returns:
            Мета-интент или пустой словарь, если не найден
        """
        return self.meta_intents.get(intent_name, {})
    
    def get_error_map(self) -> Dict[str, Any]:
        """
        Получение карты ошибок.
        
        Returns:
            Карта ошибок
        """
        return self.error_map
    
    def get_retrospection(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Получение последних записей ретроспекции.
        
        Args:
            limit: Максимальное количество записей
            
        Returns:
            Список записей ретроспекции
        """
        # Сортировка по временной метке в обратном порядке
        sorted_keys = sorted(self.retrospection_cache.keys(), 
                             key=lambda k: self.retrospection_cache[k]["timestamp"],
                             reverse=True)
        
        return [self.retrospection_cache[k] for k in sorted_keys[:limit]]


class ReflectionModule(BaseModule):
    """Модуль рефлексии SASOK."""
    
    async def initialize(self):
        """Инициализация модуля рефлексии."""
        self.logger.info("Инициализация модуля рефлексии...")
        
        # Создание движка рефлексии
        self.reflection_engine = ReflectionEngine(self.config, self.logger)
        
        # Инициализация состояния
        self.state = {
            "active": False,
            "analysis_count": 0,
            "last_reflection": None,
            "error_count": 0
        }
        
        self.logger.info("Модуль рефлексии инициализирован")
    
    async def activate(self):
        """Активация модуля рефлексии."""
        if self.active:
            self.logger.warning("Модуль рефлексии уже активен")
            return
        
        self.logger.info("Активация модуля рефлексии...")
        
        # Подписка на события для рефлексии
        await self.subscribe("emotion.state_changed", self._on_emotion_changed)
        await self.subscribe("decision.made", self._on_decision_made)
        await self.subscribe("error.occurred", self._on_error_occurred)
        await self.subscribe("interaction.completed", self._on_interaction_completed)
        
        # Запуск периодической рефлексии
        asyncio.create_task(self._periodic_reflection())
        
        self.active = True
        await self.update_state({"active": True})
        
        self.logger.info("Модуль рефлексии активирован")
    
    async def deactivate(self):
        """Деактивация модуля рефлексии."""
        if not self.active:
            self.logger.warning("Модуль рефлексии уже неактивен")
            return
        
        self.logger.info("Деактивация модуля рефлексии...")
        
        # Отписка от всех событий
        for subscription in self.subscriptions:
            await subscription.unsubscribe()
        self.subscriptions = []
        
        self.active = False
        await self.update_state({"active": False})
        
        self.logger.info("Модуль рефлексии деактивирован")
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обработка входящих данных для рефлексии.
        
        Args:
            data: Данные для обработки
            
        Returns:
            Результат рефлексии
        """
        if not self.active:
            self.logger.warning("Попытка обработки данных неактивным модулем рефлексии")
            return {"error": "Module inactive"}
        
        self.logger.info(f"Обработка данных для рефлексии: {data.get('type', 'unknown')}")
        
        try:
            # Анализ данных через движок рефлексии
            result = await self.reflection_engine.analyze(data)
            
            # Обновление состояния
            self.state["analysis_count"] += 1
            self.state["last_reflection"] = {
                "timestamp": datetime.datetime.now().isoformat(),
                "type": data.get("type", "unknown"),
                "insights_count": len(result["insights"])
            }
            
            # Публикация результатов рефлексии
            await self.publish("reflection.insight_generated", json.dumps(result))
            
            # Выполнение рекомендуемых действий
            for action in result["recommended_actions"]:
                await self.publish(
                    f"{action['target']}.request",
                    json.dumps({
                        "action": action["action"],
                        "parameters": action["parameters"],
                        "source": "reflection_module"
                    })
                )
            
            return result
        except Exception as e:
            self.logger.error(f"Ошибка при обработке данных для рефлексии: {e}")
            self.state["error_count"] += 1
            return {"error": str(e)}
    
    async def _on_emotion_changed(self, msg):
        """Обработчик события изменения эмоционального состояния."""
        data = json.loads(msg.data.decode())
        await self.process({
            "type": "emotion_analysis",
            "source": "emotion_module",
            "emotion": data
        })
    
    async def _on_decision_made(self, msg):
        """Обработчик события принятия решения."""
        data = json.loads(msg.data.decode())
        await self.process({
            "type": "decision_analysis",
            "source": "decision_module",
            "decision": data
        })
    
    async def _on_error_occurred(self, msg):
        """Обработчик события возникновения ошибки."""
        data = json.loads(msg.data.decode())
        await self.process({
            "type": "error_analysis",
            "source": data.get("source", "unknown"),
            "error": data
        })
    
    async def _on_interaction_completed(self, msg):
        """Обработчик события завершения взаимодействия."""
        data = json.loads(msg.data.decode())
        await self.process({
            "type": "interaction_analysis",
            "source": "interaction_module",
            "interaction": data
        })
    
    async def _periodic_reflection(self):
        """Периодическая рефлексия системы."""
        while self.active:
            try:
                # Сбор данных о состоянии всех модулей
                system_state = {}
                
                # Анализ системного состояния
                await self.process({
                    "type": "system_reflection",
                    "source": "reflection_module",
                    "system_state": system_state
                })