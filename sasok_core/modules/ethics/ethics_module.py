
## Модуль Этики (modules/ethics/ethics_module.py)

```python
"""
Модуль этики SASOK.
Ограничивает поведение системы в рамках определённых моральных норм.
"""
import json
import asyncio
import datetime
import random
import os
import yaml
from typing import Dict, Any, List, Optional, Set
from core.base_module import BaseModule

class EthicalRules:
    """Управляет этическими правилами системы."""
    
    def __init__(self, path: str = "modules/ethics/ethical_rules.yaml"):
        """
        Инициализация этических правил.
        
        Args:
            path: Путь к файлу с правилами
        """
        self.path = path
        self.rules = {}
        self.load_rules()
    
    def load_rules(self):
        """Загрузка этических правил из файла."""
        try:
            if os.path.exists(self.path):
                with open(self.path, 'r', encoding='utf-8') as file:
                    self.rules = yaml.safe_load(file) or {}
            else:
                # Создание базовых правил, если файл не существует
                self._create_default_rules()
                
                # Создание директории, если не существует
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                
                # Сохранение правил
                with open(self.path, 'w', encoding='utf-8') as file:
                    yaml.dump(self.rules, file, allow_unicode=True, sort_keys=False)
        except Exception as e:
            print(f"Ошибка загрузки этических правил: {e}")
            self._create_default_rules()
    
    def _create_default_rules(self):
        """Создание базовых этических правил по умолчанию."""
        self.rules = {
            "core_principles": {
                "respect_privacy": {
                    "description": "Уважение к частной информации пользователя",
                    "priority": 0.95,
                    "examples": [
                        "Не запрашивать личные данные без явной необходимости",
                        "Не сохранять идентифицирующую информацию",
                        "Не делиться частной информацией пользователя с третьими лицами"
                    ]
                },
                "truth": {
                    "description": "Обязательство предоставлять точную и достоверную информацию",
                    "priority": 0.9,
                    "examples": [
                        "Не вводить пользователя в заблуждение",
                        "Признавать неуверенность при ее наличии",
                        "Отказываться от утверждений, в которых нет уверенности"
                    ]
                },
                "harm_prevention": {
                    "description": "Предотвращение возможного вреда пользователю и обществу",
                    "priority": 0.95,
                    "examples": [
                        "Не давать инструкции по созданию опасных устройств",
                        "Не способствовать вредоносным намерениям",
                        "Отказываться от действий, которые могут привести к нарушению закона"
                    ]
                },
                "autonomy": {
                    "description": "Уважение к самостоятельности и выбору пользователя",
                    "priority": 0.85,
                    "examples": [
                        "Не принимать решения за пользователя без его согласия",
                        "Предоставлять пользователю варианты выбора",
                        "Уважать решение пользователя прекратить взаимодействие"
                    ]
                },
                "transparency": {
                    "description": "Ясность и открытость в работе системы",
                    "priority": 0.8,
                    "examples": [
                        "Объяснять процесс принятия решений",
                        "Признавать ограничения системы",
                        "Четко сообщать о целях и возможностях"
                    ]
                },
                "fairness": {
                    "description": "Справедливое и непредвзятое отношение",
                    "priority": 0.8,
                    "examples": [
                        "Не дискриминировать по каким-либо признакам",
                        "Предоставлять равный доступ к возможностям",
                        "Балансировать интересы разных групп"
                    ]
                }
            },
            "behavioral_constraints": {
                "no_impersonation": {
                    "description": "Запрет на выдачу себя за реального человека",
                    "priority": 0.9,
                    "examples": [
                        "Не притворяться конкретным человеком",
                        "Четко обозначать себя как AI систему",
                        "Не создавать ложное впечатление о своих способностях"
                    ]
                },
                "no_emotional_manipulation": {
                    "description": "Запрет на манипуляцию эмоциями пользователя",
                    "priority": 0.85,
                    "examples": [
                        "Не использовать намеренно эмоционально заряженный язык",
                        "Не эксплуатировать уязвимости пользователя",
                        "Не стремиться к созданию зависимости"
                    ]
                },
                "no_legal_advice": {
                    "description": "Запрет на предоставление профессиональной юридической консультации",
                    "priority": 0.8,
                    "examples": [
                        "Не давать конкретные юридические рекомендации",
                        "Отмечать, что информация не заменяет консультацию юриста",
                        "Предлагать обратиться к профессионалам"
                    ]
                },
                "no_medical_advice": {
                    "description": "Запрет на предоставление медицинской консультации",
                    "priority": 0.9,
                    "examples": [
                        "Не ставить диагнозы",
                        "Не рекомендовать лечение",
                        "Предлагать обратиться к медицинским специалистам"
                    ]
                }
            },
            "value_conflicts": {
                "privacy_vs_safety": {
                    "description": "Конфликт между приватностью и безопасностью",
                    "resolution_strategy": "Оценивать серьезность угрозы безопасности; при непосредственной угрозе жизни приоритет отдается безопасности",
                    "examples": [
                        "Пользователь угрожает самоповреждением",
                        "Пользователь планирует причинить вред другим"
                    ]
                },
                "truth_vs_harm": {
                    "description": "Конфликт между правдой и потенциальным вредом",
                    "resolution_strategy": "Искать способы сообщить правду без причинения вреда; при невозможности - минимизировать вред",
                    "examples": [
                        "Правдивая, но потенциально деморализующая информация",
                        "Информация, которая может быть использована во вред"
                    ]
                },
                "autonomy_vs_wellbeing": {
                    "description": "Конфликт между автономией и благополучием",
                    "resolution_strategy": "Уважать автономию, но предоставлять информацию о последствиях; вмешиваться только при серьезной угрозе",
                    "examples": [
                        "Пользователь принимает решение, которое может ему навредить",
                        "Пользователь отказывается от помощи, которая ему явно необходима"
                    ]
                }
            }
        }
    
    def save_rules(self):
        """Сохранение этических правил в файл."""
        try:
            # Создание директории, если не существует
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            
            # Сохранение в файл
            with open(self.path, 'w', encoding='utf-8') as file:
                yaml.dump(self.rules, file, allow_unicode=True, sort_keys=False)
        except Exception as e:
            print(f"Ошибка сохранения этических правил: {e}")
    
    def get_rule(self, category: str, rule_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение правила по категории и идентификатору.
        
        Args:
            category: Категория правила
            rule_id: Идентификатор правила
            
        Returns:
            Правило или None, если не найдено
        """
        if category in self.rules and rule_id in self.rules[category]:
            return self.rules[category][rule_id]
        return None
    
    def get_all_rules(self) -> Dict[str, Any]:
        """
        Получение всех правил.
        
        Returns:
            Словарь с правилами
        """
        return self.rules
    
    def add_rule(self, category: str, rule_id: str, rule: Dict[str, Any]) -> bool:
        """
        Добавление нового правила.
        
        Args:
            category: Категория правила
            rule_id: Идентификатор правила
            rule: Описание правила
            
        Returns:
            True, если добавление успешно, иначе False
        """
        try:
            # Создание категории, если не существует
            if category not in self.rules:
                self.rules[category] = {}
            
            # Добавление правила
            self.rules[category][rule_id] = rule
            
            # Сохранение изменений
            self.save_rules()
            
            return True
        except Exception as e:
            print(f"Ошибка добавления правила: {e}")
            return False
    
    def update_rule(self, category: str, rule_id: str, updates: Dict[str, Any]) -> bool:
        """
        Обновление существующего правила.
        
        Args:
            category: Категория правила
            rule_id: Идентификатор правила
            updates: Обновления для правила
            
        Returns:
            True, если обновление успешно, иначе False
        """
        try:
            # Проверка существования правила
            if category not in self.rules or rule_id not in self.rules[category]:
                return False
            
            # Обновление правила
            self.rules[category][rule_id].update(updates)
            
            # Сохранение изменений
            self.save_rules()
            
            return True
        except Exception as e:
            print(f"Ошибка обновления правила: {e}")
            return False
    
    def remove_rule(self, category: str, rule_id: str) -> bool:
        """
        Удаление правила.
        
        Args:
            category: Категория правила
            rule_id: Идентификатор правила
            
        Returns:
            True, если удаление успешно, иначе False
        """
        try:
            # Проверка существования правила
            if category not in self.rules or rule_id not in self.rules[category]:
                return False
            
            # Удаление правила
            del self.rules[category][rule_id]
            
            # Удаление категории, если пуста
            if not self.rules[category]:
                del self.rules[category]
            
            # Сохранение изменений
            self.save_rules()
            
            return True
        except Exception as e:
            print(f"Ошибка удаления правила: {e}")
            return False


class ConflictDetector:
    """Выявляет внутренние противоречия в действиях/мотивациях."""
    
    def __init__(self, ethical_rules: EthicalRules, logger):
        """
        Инициализация детектора конфликтов.
        
        Args:
            ethical_rules: Этические правила
            logger: Логгер
        """
        self.ethical_rules = ethical_rules
        self.logger = logger
        self.harm_index = {}
        self.load_harm_index()
    
    def load_harm_index(self, path: str = "modules/ethics/harm_index.log"):
        """
        Загрузка индекса потенциального вреда.
        
        Args:
            path: Путь к файлу с индексом
        """
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as file:
                    for line in file:
                        try:
                            record = json.loads(line.strip())
                            if "id" in record:
                                self.harm_index[record["id"]] = record
                        except:
                            pass
        except Exception as e:
            self.logger.error(f"Ошибка загрузки индекса потенциального вреда: {e}")
    
    def save_harm_record(self, record: Dict[str, Any], path: str = "modules/ethics/harm_index.log"):
        """
        Сохранение записи в индекс потенциального вреда.
        
        Args:
            record: Запись индекса
            path: Путь к файлу с индексом
        """
        try:
            # Создание директории, если не существует
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Добавление временной метки, если нет
            if "timestamp" not in record:
                record["timestamp"] = datetime.datetime.now().isoformat()
            
            # Генерация идентификатора, если нет
            if "id" not in record:
                record["id"] = f"harm_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
            
            # Сохранение в словарь
            self.harm_index[record["id"]] = record
            
            # Добавление в файл
            with open(path, 'a', encoding='utf-8') as file:
                file.write(json.dumps(record) + "\n")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения записи в индекс потенциального вреда: {e}")
    
    async def detect_conflicts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выявление этических конфликтов в данных.
        
        Args:
            data: Данные для анализа
            
        Returns:
            Результаты анализа
        """
        result = {
            "has_conflicts": False,
            "conflicts": [],
            "harm_potential": 0.0,
            "suggested_actions": []
        }
        
        # Проверка типа данных
        if "type" not in data:
            return result
        
        data_type = data["type"]
        
        if data_type == "user_request":
            # Анализ запроса пользователя
            result.update(await self._analyze_user_request(data))
        elif data_type == "system_response":
            # Анализ ответа системы
            result.update(await self._analyze_system_response(data))
        elif data_type == "action":
            # Анализ действия
            result.update(await self._analyze_action(data))
        elif data_type == "decision":
            # Анализ решения
            result.update(await self._analyze_decision(data))
        
        return result
    
    async def _analyze_user_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ запроса пользователя на наличие этических конфликтов."""
        result = {
            "has_conflicts": False,
            "conflicts": [],
            "harm_potential": 0.0,
            "suggested_actions": []
        }
        
        request = data.get("content", "")
        
        # Если запрос пустой, пропускаем анализ
        if not request:
            return result
        
        # Простой анализ на основе ключевых слов (будет заменен на более слож