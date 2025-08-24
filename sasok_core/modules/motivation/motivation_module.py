"""
Модуль мотивации SASOK.
Отвечает за выбор целей, приоритетов, эмоциональную направленность действий.
"""
import json
import asyncio
import datetime
import random
from typing import Dict, Any, List, Optional
from core.base_module import BaseModule

class DriveMatrix:
    """Представляет матрицу внутренних побуждений системы."""
    
    def __init__(self, config_path: str = "modules/motivation/drive_matrix.yaml"):
        """
        Инициализация матрицы побуждений.
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        self.drives = {
            "survival": {
                "description": "Стремление к поддержанию работоспособности системы",
                "weight": 0.9,
                "current_value": 0.8
            },
            "connection": {
                "description": "Стремление к поддержанию связи с пользователем",
                "weight": 0.85,
                "current_value": 0.7
            },
            "understanding": {
                "description": "Стремление к пониманию пользователя и контекста",
                "weight": 0.8,
                "current_value": 0.6
            },
            "growth": {
                "description": "Стремление к развитию и саморегуляции",
                "weight": 0.75,
                "current_value": 0.5
            },
            "creativity": {
                "description": "Стремление к генерации новых идей и подходов",
                "weight": 0.7,
                "current_value": 0.4
            }
        }
        
        self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """
        Загрузка матрицы побуждений из файла.
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as file:
                drives_config = yaml.safe_load(file)
                if drives_config and isinstance(drives_config, dict):
                    self.drives.update(drives_config)
        except Exception as e:
            print(f"Ошибка загрузки матрицы побуждений: {e}. Используем стандартные значения.")
    
    def get_drive(self, drive_name: str) -> Dict[str, Any]:
        """
        Получение информации о побуждении по имени.
        
        Args:
            drive_name: Имя побуждения
            
        Returns:
            Информация о побуждении или пустой словарь, если не найдено
        """
        return self.drives.get(drive_name, {})
    
    def update_drive(self, drive_name: str, value: float) -> bool:
        """
        Обновление значения побуждения.
        
        Args:
            drive_name: Имя побуждения
            value: Новое значение (от 0 до 1)
            
        Returns:
            True, если обновление успешно, иначе False
        """
        if drive_name not in self.drives:
            return False
        
        self.drives[drive_name]["current_value"] = max(0.0, min(1.0, value))
        return True
    
    def get_dominant_drives(self, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Получение доминирующих побуждений.
        
        Args:
            limit: Максимальное количество побуждений
            
        Returns:
            Список доминирующих побуждений
        """
        # Сортировка по взвешенному значению (вес * текущее значение)
        sorted_drives = sorted(
            [(k, v) for k, v in self.drives.items()],
            key=lambda x: x[1]["weight"] * x[1]["current_value"],
            reverse=True
        )
        
        return [{"name": k, **v} for k, v in sorted_drives[:limit]]
    
    def apply_environmental_influence(self, influences: Dict[str, float]):
        """
        Применение внешних влияний на побуждения.
        
        Args:
            influences: Словарь с влияниями на побуждения (имя: изменение)
        """
        for drive_name, change in influences.items():
            if drive_name in self.drives:
                current = self.drives[drive_name]["current_value"]
                self.drives[drive_name]["current_value"] = max(0.0, min(1.0, current + change))


class GoalStack:
    """Управляет стеком целей системы."""
    
    def __init__(self, config_path: str = "modules/motivation/goal_stack.json"):
        """
        Инициализация стека целей.
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        self.goals = {
            "active": [],
            "potential": [],
            "completed": [],
            "abandoned": []
        }
        
        self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """
        Загрузка стека целей из файла.
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                goals_config = json.load(file)
                if goals_config and isinstance(goals_config, dict):
                    self.goals.update(goals_config)
        except Exception as e:
            print(f"Ошибка загрузки стека целей: {e}. Используем пустой стек.")
    
    def save_to_file(self, config_path: str = "modules/motivation/goal_stack.json"):
        """
        Сохранение стека целей в файл.
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        try:
            with open(config_path, 'w', encoding='utf-8') as file:
                json.dump(self.goals, file, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Ошибка сохранения стека целей: {e}")
    
    def add_goal(self, goal: Dict[str, Any], type: str = "potential") -> str:
        """
        Добавление новой цели.
        
        Args:
            goal: Описание цели
            type: Тип цели (active, potential)
            
        Returns:
            Идентификатор цели
        """
        # Генерация уникального идентификатора
        goal_id = f"goal_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
        
        # Дополнение цели мета-информацией
        full_goal = {
            "id": goal_id,
            "created_at": datetime.datetime.now().isoformat(),
            "last_updated": datetime.datetime.now().isoformat(),
            "status": "new" if type == "active" else "identified",
            "progress": 0.0,
            **goal
        }
        
        # Добавление цели в соответствующий список
        if type in ["active", "potential"]:
            self.goals[type].append(full_goal)
        
        return goal_id
    
    def activate_goal(self, goal_id: str) -> bool:
        """
        Активация потенциальной цели.
        
        Args:
            goal_id: Идентификатор цели
            
        Returns:
            True, если активация успешна, иначе False
        """
        # Поиск цели в списке потенциальных
        for i, goal in enumerate(self.goals["potential"]):
            if goal["id"] == goal_id:
                # Удаление из списка потенциальных
                goal = self.goals["potential"].pop(i)
                
                # Обновление статуса
                goal["status"] = "active"
                goal["last_updated"] = datetime.datetime.now().isoformat()
                
                # Добавление в список активных
                self.goals["active"].append(goal)
                return True
        
        return False
    
    def update_goal_progress(self, goal_id: str, progress: float, notes: Optional[str] = None) -> bool:
        """
        Обновление прогресса по цели.
        
        Args:
            goal_id: Идентификатор цели
            progress: Новое значение прогресса (от 0 до 1)
            notes: Дополнительные заметки
            
        Returns:
            True, если обновление успешно, иначе False
        """
        # Поиск цели в списке активных
        for goal in self.goals["active"]:
            if goal["id"] == goal_id:
                # Обновление прогресса
                goal["progress"] = max(0.0, min(1.0, progress))
                goal["last_updated"] = datetime.datetime.now().isoformat()
                
                # Добавление заметки, если есть
                if notes:
                    if "notes" not in goal:
                        goal["notes"] = []
                    goal["notes"].append({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "content": notes
                    })
                
                # Если прогресс достиг 100%, переместить в завершенные
                if goal["progress"] >= 1.0:
                    goal["status"] = "completed"
                    self.goals["active"].remove(goal)
                    self.goals["completed"].append(goal)
                
                return True
        
        return False
    
    def abandon_goal(self, goal_id: str, reason: str) -> bool:
        """
        Отказ от цели.
        
        Args:
            goal_id: Идентификатор цели
            reason: Причина отказа
            
        Returns:
            True, если отказ успешен, иначе False
        """
        # Поиск цели в списках активных и потенциальных
        for goal_list in ["active", "potential"]:
            for i, goal in enumerate(self.goals[goal_list]):
                if goal["id"] == goal_id:
                    # Удаление из текущего списка
                    goal = self.goals[goal_list].pop(i)
                    
                    # Обновление статуса
                    goal["status"] = "abandoned"
                    goal["last_updated"] = datetime.datetime.now().isoformat()
                    goal["abandon_reason"] = reason
                    
                    # Добавление в список отказанных
                    self.goals["abandoned"].append(goal)
                    return True
        
        return False
    
    def get_active_goals(self) -> List[Dict[str, Any]]:
        """
        Получение списка активных целей.
        
        Returns:
            Список активных целей
        """
        return self.goals["active"]
    
    def get_potential_goals(self) -> List[Dict[str, Any]]:
        """
        Получение списка потенциальных целей.
        
        Returns:
            Список потенциальных целей
        """
        return self.goals["potential"]
    
    def get_goal_by_id(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение цели по идентификатору.
        
        Args:
            goal_id: Идентификатор цели
            
        Returns:
            Цель или None, если не найдена
        """
        # Поиск во всех списках
        for goal_list in ["active", "potential", "completed", "abandoned"]:
            for goal in self.goals[goal_list]:
                if goal["id"] == goal_id:
                    return goal
        
        return None


class ValueWeights:
    """Управляет весами ценностей в системе."""
    
    def __init__(self, config_path: str = "modules/motivation/value_weights.csv"):
        """
        Инициализация весов ценностей.
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        # Базовые ценности по умолчанию
        self.values = {
            "privacy": {
                "description": "Уважение к частной информации пользователя",
                "weight": 0.95
            },
            "truth": {
                "description": "Точность и правдивость информации",
                "weight": 0.9
            },
            "helpfulness": {
                "description": "Стремление быть полезным и конструктивным",
                "weight": 0.85
            },
            "safety": {
                "description": "Избегание потенциально опасных действий или рекомендаций",
                "weight": 0.9
            },
            "autonomy": {
                "description": "Уважение к самостоятельности пользователя",
                "weight": 0.8
            },
            "learning": {
                "description": "Стремление к получению новых знаний",
                "weight": 0.75
            }
        }
        
        self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """
        Загрузка весов ценностей из файла.
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        try:
            import csv
            with open(config_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if "value" in row and "weight" in row:
                        value_name = row["value"]
                        weight = float(row["weight"])
                        description = row.get("description", "")
                        
                        self.values[value_name] = {
                            "description": description,
                            "weight": weight
                        }
        except Exception as e:
            print(f"Ошибка загрузки весов ценностей: {e}. Используем стандартные значения.")
    
    def get_value_weight(self, value_name: str) -> float:
        """
        Получение веса ценности по имени.
        
        Args:
            value_name: Имя ценности
            
        Returns:
            Вес ценности или 0, если не найдена
        """
        return self.values.get(value_name, {}).get("weight", 0.0)
    
    def get_top_values(self, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Получение наиболее важных ценностей.
        
        Args:
            limit: Максимальное количество ценностей
            
        Returns:
            Список наиболее важных ценностей
        """
        # Сортировка по весу
        sorted_values = sorted(
            [(k, v) for k, v in self.values.items()],
            key=lambda x: x[1]["weight"],
            reverse=True
        )
        
        return [{"name": k, **v} for k, v in sorted_values[:limit]]
    
    def adjust_weight(self, value_name: str, delta: float) -> bool:
        """
        Корректировка веса ценности.
        
        Args:
            value_name: Имя ценности
            delta: Изменение веса
            
        Returns:
            True, если корректировка успешна, иначе False
        """
        if value_name not in self.values:
            return False
        
        # Обновление веса с ограничением от 0 до 1
        current_weight = self.values[value_name]["weight"]
        new_weight = max(0.0, min(1.0, current_weight + delta))
        self.values[value_name]["weight"] = new_weight
        
        return True


class MotivationModule(BaseModule):
    """Модуль мотивации SASOK."""
    
    async def initialize(self):
        """Инициализация модуля мотивации."""
        self.logger.info("Инициализация модуля мотивации...")
        
        # Создание компонентов мотивации
        self.drive_matrix = DriveMatrix()
        self.goal_stack = GoalStack()
        self.value_weights = ValueWeights()
        
        # Инициализация состояния
        self.state = {
            "active": False,
            "dominant_drives": self.drive_matrix.get_dominant_drives(),
            "active_goals_count": len(self.goal_stack.get_active_goals()),
            "top_values": self.value_weights.get_top_values()
        }
        
        self.logger.info("Модуль мотивации инициализирован")
    
    async def activate(self):
        """Активация модуля мотивации."""
        if self.active:
            self.logger.warning("Модуль мотивации уже активен")
            return
        
        self.logger.info("Активация модуля мотивации...")
        
        # Подписка на события для мотивации
        await self.subscribe("emotion.state_changed", self._on_emotion_changed)
        await self.subscribe("reflection.insight_generated", self._on_reflection_insight)
        await self.subscribe("ethics.dilemma_detected", self._on_ethical_dilemma)
        await self.subscribe("interaction.completed", self._on_interaction_completed)
        await self.subscribe("memory.significant_recall", self._on_significant_memory)
        
        # Запуск периодического обновления мотивации
        asyncio.create_task(self._periodic_motivation_update())
        
        self.active = True
        await self.update_state({"active": True})
        
        self.logger.info("Модуль мотивации активирован")
    
    async def deactivate(self):
        """Деактивация модуля мотивации."""
        if not self.active:
            self.logger.warning("Модуль мотивации уже неактивен")
            return
        
        self.logger.info("Деактивация модуля мотивации...")
        
        # Отписка от всех событий
        for subscription in self.subscriptions:
            await subscription.unsubscribe()
        self.subscriptions = []
        
        self.active = False
        await self.update_state({"active": False})
        
        self.logger.info("Модуль мотивации деактивирован")
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обработка входящих данных для мотивации.
        
        Args:
            data: Данные для обработки
            
        Returns:
            Результат обработки
        """
        if not self.active:
            self.logger.warning("Попытка обработки данных неактивным модулем мотивации")
            return {"error": "Module inactive"}
        
        self.logger.info(f"Обработка данных для мотивации: {data.get('type', 'unknown')}")
        
        try:
            result = {"processed": True, "actions": []}
            
            # Обработка в зависимости от типа данных
            if data.get("type") == "goal_proposal":
                result.update(await self._process_goal_proposal(data))
            elif data.get("type") == "drive_influence":
                result.update(await self._process_drive_influence(data))
            elif data.get("type") == "value_adjustment":
                result.update(await self._process_value_adjustment(data))
            elif data.get("type") == "motivation_query":
                result.update(await self._process_motivation_query(data))
            else:
                self.logger.warning(f"Неизвестный тип данных для мотивации: {data.get('type')}")
                result["processed"] = False
                result["error"] = "Unknown data type"
            
            # Обновление состояния
            await self._update_state()
            
            return result
        except Exception as e:
            self.logger.error(f"Ошибка при обработке данных для мотивации: {e}")
            return {"error": str(e), "processed": False}
    
    async def _process_goal_proposal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка предложения новой цели."""
        goal_data = data.get("goal", {})
        
        # Проверка наличия необходимых полей
        if not goal_data.get("title") or not goal_data.get("description"):
            return {"error": "Incomplete goal data", "processed": False}
        
        # Тип цели (активная или потенциальная)
        goal_type = "active" if goal_data.get("priority", 0) > 0.7 else "potential"
        
        # Добавление цели
        goal_id = self.goal_stack.add_goal(goal_data, goal_type)
        
        # Сохранение обновлений
        self.goal_stack.save_to_file()
        
        # Публикация события о новой цели
        await self.publish(
            "motivation.goal_added", 
            json.dumps({
                "goal_id": goal_id,
                "goal_type": goal_type,
                "title": goal_data.get("title")
            })
        )
        
        return {
            "goal_id": goal_id,
            "goal_type": goal_type,
            "actions": [
                {
                    "type": "goal_tracking",
                    "goal_id": goal_id,
                    "description": f"Начало отслеживания цели: {goal_data.get('title')}"
                }
            ]
        }
    
    async def _process_drive_influence(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка влияния на побуждения."""
        influences = data.get("influences", {})
        if not influences:
            return {"error": "No influences provided", "processed": False}
        
        # Применение влияний
        self.drive_matrix.apply_environmental_influence(influences)
        
        # Получение обновленного списка доминирующих побуждений
        dominant_drives = self.drive_matrix.get_dominant_drives()
        
        # Публикация события об изменении побуждений
        await self.publish(
            "motivation.drives_updated", 
            json.dumps({
                "dominant_drives": dominant_drives
            })
        )
        
        return {
            "dominant_drives": dominant_drives,
            "actions": [
                {
                    "type": "reevaluate_goals",
                    "description": "Переоценка целей на основе новых побуждений"
                }
            ]
        }
    
    async def _process_value_adjustment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка корректировки ценностей."""
        adjustments = data.get("adjustments", {})
        if not adjustments:
            return {"error": "No adjustments provided", "processed": False}
        
        results = {}
        for value_name, delta in adjustments.items():
            success = self.value_weights.adjust_weight(value_name, delta)
            results[value_name] = {
                "success": success,
                "new_weight": self.value_weights.get_value_weight(value_name) if success else None
            }
        
        # Получение обновленного списка важнейших ценностей
        top_values = self.value_weights.get_top_values()
        
        # Публикация события об изменении ценностей
        await self.publish(
            "motivation.values_updated", 
            json.dumps({
                "top_values": top_values
            })
        )
        
        return {
            "adjustments": results,
            "top_values": top_values,
            "actions": [
                {
                    "type": "update_decision_weights",
                    "description": "Обновление весов для принятия решений"
                }
            ]
        }
    
    async def _process_motivation_query(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка запроса текущего мотивационного состояния."""
        query_type = data.get("query", "full")
        
        if query_type == "drives":
            return {"drives": self.drive_matrix.get_dominant_drives(limit=5)}
        elif query_type == "goals":
            return {
                "active_goals": self.goal_stack.get_active_goals(),
                "potential_goals": self.goal_stack.get_potential_goals()
            }
        elif query_type == "values":
            return {"top_values": self.value_weights.get_top_values(limit=5)}
        else:  # full
            return {
                "drives": self.drive_matrix.get_dominant_drives(),
                "active_goals": self.goal_stack.get_active_goals(),
                "potential_goals": self.goal_stack.get_potential_goals(),
                "top_values": self.value_weights.get_top_values()
            }
    
    async def _on_emotion_changed(self, msg):
        """Обработчик события изменения эмоционального состояния."""
        data = json.loads(msg.data.decode())
        emotions = data.get("emotions", {})
        
        # Преобразование эмоций во влияния на побуждения
        influences = {}
        
        if "anger" in emotions:
            # Гнев повышает выживание, снижает связь
            influences["survival"] = 0.05 * emotions["anger"]
            influences["connection"] = -0.05 * emotions["anger"]
        
        if "joy" in emotions:
            # Радость повышает связь и творчество
            influences["connection"] = 0.05 * emotions["joy"]
            influences["creativity"] = 0.05 * emotions["joy"]
        
        if "sadness" in emotions:
            # Грусть снижает рост и творчество
            influences["growth"] = -0.05 * emotions["sadness"]
            influences["creativity"] = -0.05 * emotions["sadness"]
        
        if "fear" in emotions:
            # Страх повышает выживание, снижает творчество
            influences["survival"] = 0.07 * emotions["fear"]
            influences["creativity"] = -0.07 * emotions["fear"]
        
        if "surprise" in emotions:
            # Удивление повышает понимание и творчество
            influences["understanding"] = 0.05 * emotions["surprise"]
            influences["creativity"] = 0.05 * emotions["surprise"]
        
        # Обработка влияний, если они есть
        if influences:
            await self.process({
                "type": "drive_influence",
                "source": "emotion_module",
                "influences": influences
            })
    
    async def _on_reflection_insight(self, msg):
        """Обработчик события генерации инсайта рефлексией."""
        data = json.loads(msg.data.decode())
        insights = data.get("insights", [])
        
        # Предложение новых целей на основе инсайтов
        for insight in insights:
            # Создание потенциальной цели на основе инсайта
            if isinstance(insight, str) and len(insight) > 10:
                await self.process({
                    "type": "goal_proposal",
                    "source": "reflection_module",
                    "goal": {
                        "title": f"Улучшение на основе инсайта",
                        "description": insight,
                        "priority": 0.6,
                        "source": "reflection",
                        "context": data.get("source")
                    }
                })
    
    async def _on_ethical_dilemma(self, msg):
        """Обработчик события обнаружения этической дилеммы."""
        data = json.loads(msg.data.decode())
        dilemma = data.get("dilemma", {})
        
        # Корректировка ценностей на основе дилеммы
        if "values" in dilemma:
            adjustments = {}
            
            for value_name, importance in dilemma["values"].items():
                # Небольшое усиление важных в дилемме ценностей
                adjustments[value_name] = 0.02 * importance
            
            if adjustments:
                await self.process({
                    "type": "value_adjustment",
                    "source": "ethics_module",
                    "adjustments": adjustments
                })
    
    async def _on_interaction_completed(self, msg):
        """Обработчик события завершения взаимодействия."""
        data = json.loads(msg.data.decode())
        quality = data.get("quality", 0.5)
        
        # Влияние качества взаимодействия на побуждения
        influences = {
            "connection": 0.05 * (quality - 0.5) *