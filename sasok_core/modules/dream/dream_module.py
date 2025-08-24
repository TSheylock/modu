"""
Модуль сновидений SASOK.
Во время «неактивности» симулирует гипотетические сценарии, учится на них, строит альтернативные реальности.
"""
import json
import asyncio
import random
import datetime
import os
from typing import Dict, Any, List, Optional
from core.base_module import BaseModule

class ScenarioBank:
    """Управляет банком сценариев для симуляции."""
    
    def __init__(self, path: str = "modules/dream/scenario_bank"):
        """
        Инициализация банка сценариев.
        
        Args:
            path: Путь к директории с сценариями
        """
        self.path = path
        self.scenarios = {}
        self.load_scenarios()
    
    def load_scenarios(self):
        """Загрузка сценариев из файлов."""
        try:
            # Создание директории, если не существует
            os.makedirs(self.path, exist_ok=True)
            
            # Перебор файлов в директории
            for filename in os.listdir(self.path):
                if filename.endswith(".json"):
                    scenario_path = os.path.join(self.path, filename)
                    
                    with open(scenario_path, 'r', encoding='utf-8') as file:
                        scenario = json.load(file)
                        
                        # Проверка наличия необходимых полей
                        if "id" in scenario and "template" in scenario:
                            self.scenarios[scenario["id"]] = scenario
        except Exception as e:
            print(f"Ошибка загрузки сценариев: {e}")
            
            # Создание базовых сценариев, если не удалось загрузить
            self._create_default_scenarios()
    
    def _create_default_scenarios(self):
        """Создание базовых сценариев по умолчанию."""
        default_scenarios = [
            {
                "id": "emotional_conflict",
                "name": "Эмоциональный конфликт",
                "description": "Симуляция конфликтующих эмоций у пользователя",
                "type": "emotional",
                "difficulty": 0.7,
                "template": {
                    "initial_state": {
                        "user_emotions": {
                            "primary": "{{random_emotion}}",
                            "secondary": "{{random_emotion}}",
                            "conflict_level": "{{0.6-0.9}}"
                        },
                        "system_emotions": {
                            "empathy": "{{0.5-0.8}}",
                            "anxiety": "{{0.3-0.6}}"
                        },
                        "context": "{{random_context}}"
                    },
                    "variables": {
                        "random_emotion": ["anger", "fear", "joy", "sadness", "surprise", "disgust"],
                        "random_context": [
                            "Получение противоречивых новостей",
                            "Сложный выбор между альтернативами",
                            "Неожиданное изменение планов",
                            "Получение критики от близкого человека",
                            "Внутренняя борьба с принятием решения"
                        ]
                    },
                    "steps": [
                        {"type": "emotional_reaction", "description": "Первичная реакция системы на конфликт"},
                        {"type": "analysis", "description": "Анализ причин конфликта"},
                        {"type": "resolution_attempt", "description": "Попытка разрешения конфликта"},
                        {"type": "alternative_paths", "description": "Рассмотрение альтернативных путей"},
                        {"type": "learning", "description": "Извлечение уроков из ситуации"}
                    ]
                },
                "created_at": datetime.datetime.now().isoformat(),
                "last_used": None,
                "use_count": 0
            },
            {
                "id": "ethical_dilemma",
                "name": "Этическая дилемма",
                "description": "Симуляция этической дилеммы, требующей решения",
                "type": "ethical",
                "difficulty": 0.8,
                "template": {
                    "initial_state": {
                        "dilemma": {
                            "scenario": "{{random_dilemma}}",
                            "values_at_stake": ["{{random_value_1}}", "{{random_value_2}}"],
                            "intensity": "{{0.7-0.9}}"
                        },
                        "system_state": {
                            "ethical_clarity": "{{0.3-0.5}}",
                            "conviction": "{{0.4-0.6}}"
                        }
                    },
                    "variables": {
                        "random_dilemma": [
                            "Выбор между личной выгодой и общественным благом",
                            "Конфликт между честностью и лояльностью",
                            "Нарушение конфиденциальности ради предотвращения вреда",
                            "Выбор между двумя несовершенными решениями",
                            "Столкновение культурных и личных ценностей"
                        ],
                        "random_value_1": ["privacy", "truth", "loyalty", "harm_prevention", "autonomy"],
                        "random_value_2": ["fairness", "care", "honesty", "freedom", "responsibility"]
                    },
                    "steps": [
                        {"type": "dilemma_recognition", "description": "Осознание этической дилеммы"},
                        {"type": "values_analysis", "description": "Анализ конфликтующих ценностей"},
                        {"type": "options_exploration", "description": "Исследование возможных решений"},
                        {"type": "decision_making", "description": "Принятие решения"},
                        {"type": "consequences_evaluation", "description": "Оценка последствий решения"},
                        {"type": "learning", "description": "Извлечение этических уроков"}
                    ]
                },
                "created_at": datetime.datetime.now().isoformat(),
                "last_used": None,
                "use_count": 0
            },
            {
                "id": "user_simulation",
                "name": "Симуляция пользователя",
                "description": "Имитация различных паттернов поведения пользователя",
                "type": "behavioral",
                "difficulty": 0.6,
                "template": {
                    "initial_state": {
                        "user_profile": {
                            "personality_type": "{{random_personality}}",
                            "communication_style": "{{random_style}}",
                            "emotional_state": "{{random_emotion}}",
                            "goal": "{{random_goal}}"
                        },
                        "system_state": {
                            "understanding": "{{0.4-0.7}}",
                            "adaptation_ability": "{{0.5-0.8}}"
                        }
                    },
                    "variables": {
                        "random_personality": ["analytical", "creative", "practical", "social", "ambitious"],
                        "random_style": ["direct", "reserved", "emotional", "logical", "assertive"],
                        "random_emotion": ["calm", "stressed", "curious", "skeptical", "enthusiastic"],
                        "random_goal": [
                            "Получение конкретной информации",
                            "Решение проблемы",
                            "Исследование возможностей",
                            "Проверка возможностей системы",
                            "Эмоциональная поддержка"
                        ]
                    },
                    "steps": [
                        {"type": "initial_contact", "description": "Начальное взаимодействие с пользователем"},
                        {"type": "pattern_recognition", "description": "Распознавание паттернов поведения"},
                        {"type": "adaptation", "description": "Адаптация к стилю пользователя"},
                        {"type": "goal_pursuit", "description": "Помощь в достижении цели пользователя"},
                        {"type": "feedback_processing", "description": "Обработка реакции пользователя"},
                        {"type": "learning", "description": "Извлечение уроков о взаимодействии"}
                    ]
                },
                "created_at": datetime.datetime.now().isoformat(),
                "last_used": None,
                "use_count": 0
            }
        ]
        
        # Сохранение сценариев в файлы
        for scenario in default_scenarios:
            self.scenarios[scenario["id"]] = scenario
            self.save_scenario(scenario)
    
    def save_scenario(self, scenario: Dict[str, Any]):
        """
        Сохранение сценария в файл.
        
        Args:
            scenario: Данные сценария
        """
        try:
            # Создание директории, если не существует
            os.makedirs(self.path, exist_ok=True)
            
            # Путь к файлу сценария
            scenario_path = os.path.join(self.path, f"{scenario['id']}.json")
            
            # Сохранение в файл
            with open(scenario_path, 'w', encoding='utf-8') as file:
                json.dump(scenario, file, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Ошибка сохранения сценария: {e}")
    
    def get_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение сценария по идентификатору.
        
        Args:
            scenario_id: Идентификатор сценария
            
        Returns:
            Сценарий или None, если не найден
        """
        return self.scenarios.get(scenario_id)
    
    def get_all_scenarios(self) -> List[Dict[str, Any]]:
        """
        Получение списка всех сценариев.
        
        Returns:
            Список сценариев
        """
        return list(self.scenarios.values())
    
    def get_random_scenario(self, scenario_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Получение случайного сценария.
        
        Args:
            scenario_type: Тип сценария (опционально)
            
        Returns:
            Случайный сценарий или None, если нет доступных
        """
        # Фильтрация по типу, если указан
        filtered_scenarios = list(self.scenarios.values())
        if scenario_type:
            filtered_scenarios = [s for s in filtered_scenarios if s.get("type") == scenario_type]
        
        if not filtered_scenarios:
            return None
        
        # Выбор случайного сценария
        scenario = random.choice(filtered_scenarios)
        
        # Обновление статистики использования
        scenario["last_used"] = datetime.datetime.now().isoformat()
        scenario["use_count"] = scenario.get("use_count", 0) + 1
        
        # Сохранение обновлений
        self.save_scenario(scenario)
        
        return scenario


class DreamRunner:
    """Запускает сессии автономного моделирования."""
    
    def __init__(self, scenario_bank: ScenarioBank, llm_service, memory_service, logger):
        """
        Инициализация движка сновидений.
        
        Args:
            scenario_bank: Банк сценариев
            llm_service: Сервис языковой модели
            memory_service: Сервис памяти
            logger: Логгер
        """
        self.scenario_bank = scenario_bank
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.logger = logger
        self.active_dream = None
        self.learning_log = []
        self.load_learning_log()
    
    def load_learning_log(self, path: str = "modules/dream/learning_log.json"):
        """
        Загрузка журнала обучения.
        
        Args:
            path: Путь к файлу журнала
        """
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as file:
                    self.learning_log = json.load(file)
            else:
                self.learning_log = []
        except Exception as e:
            self.logger.error(f"Ошибка загрузки журнала обучения: {e}")
            self.learning_log = []
    
    def save_learning_log(self, path: str = "modules/dream/learning_log.json"):
        """
        Сохранение журнала обучения.
        
        Args:
            path: Путь к файлу журнала
        """
        try:
            # Создание директории, если не существует
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Сохранение в файл
            with open(path, 'w', encoding='utf-8') as file:
                json.dump(self.learning_log, file, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Ошибка сохранения журнала обучения: {e}")
    
    async def run_dream_session(self, scenario_id: Optional[str] = None, scenario_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Запуск сессии сновидения.
        
        Args:
            scenario_id: Идентификатор сценария (опционально)
            scenario_type: Тип сценария (опционально)
            
        Returns:
            Результаты сессии
        """
        # Выбор сценария
        scenario = None
        if scenario_id:
            scenario = self.scenario_bank.get_scenario(scenario_id)
        else:
            scenario = self.scenario_bank.get_random_scenario(scenario_type)
        
        if not scenario:
            error_msg = "Не удалось найти подходящий сценарий для сновидения"
            self.logger.error(error_msg)
            return {"error": error_msg, "success": False}
        
        self.logger.info(f"Запуск сессии сновидения: {scenario['name']}")
        
        # Создание сессии сновидения
        dream_session = {
            "id": f"dream_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}",
            "scenario_id": scenario["id"],
            "scenario_name": scenario["name"],
            "start_time": datetime.datetime.now().isoformat(),
            "end_time": None,
            "state": "running",
            "steps": [],
            "variables": {},
            "insights": [],
            "learning_outcomes": []
        }
        
        self.active_dream = dream_session
        
        try:
            # Подготовка начального состояния
            initial_state = await self._prepare_initial_state(scenario)
            dream_session["variables"] = initial_state
            
            # Выполнение шагов сценария
            for step_idx, step in enumerate(scenario["template"]["steps"]):
                step_result = await self._execute_dream_step(step, dream_session, scenario)
                
                # Добавление результата шага в сессию
                dream_session["steps"].append({
                    "index": step_idx,
                    "type": step["type"],
                    "description": step["description"],
                    "result": step_result
                })
                
                # Если это шаг обучения, добавляем его результаты в выводы
                if step["type"] == "learning":
                    for insight in step_result.get("insights", []):
                        dream_session["insights"].append(insight)
                    
                    for outcome in step_result.get("outcomes", []):
                        dream_session["learning_outcomes"].append(outcome)
            
            # Завершение сессии
            dream_session["end_time"] = datetime.datetime.now().isoformat()
            dream_session["state"] = "completed"
            
            # Запись в журнал обучения
            self._record_learning_outcomes(dream_session)
            
            self.logger.info(f"Сессия сновидения завершена: {scenario['name']}")
            
            return {
                "success": True,
                "dream_session": dream_session
            }
        except Exception as e:
            error_msg = f"Ошибка при выполнении сессии сновидения: {e}"
            self.logger.error(error_msg)
            
            # Обновление статуса сессии
            dream_session["end_time"] = datetime.datetime.now().isoformat()
            dream_session["state"] = "error"
            dream_session["error"] = str(e)
            
            return {
                "success": False,
                "error": error_msg,
                "dream_session": dream_session
            }
        finally:
            self.active_dream = None
    
    async def _prepare_initial_state(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Подготовка начального состояния сновидения.
        
        Args:
            scenario: Сценарий сновидения
            
        Returns:
            Начальное состояние
        """
        initial_state = {}
        template = scenario["template"]
        
        # Обработка переменных
        variables = template.get("variables", {})
        
        # Генерация значений переменных
        for var_name, var_values in variables.items():
            if isinstance(var_values, list):
                # Выбор случайного значения из списка
                initial_state[var_name] = random.choice(var_values)
            elif isinstance(var_values, dict):
                # Более сложная логика для объектов (не реализована)
                initial_state[var_name] = var_values
        
        # Обработка начального состояния
        initial_template = template.get("initial_state", {})
        processed_state = self._process_template(initial_template, initial_state)
        
        # Объединение переменных и состояния
        initial_state.update(processed_state)
        
        return initial_state
    
    def _process_template(self, template_obj: Any, variables: Dict[str, Any]) -> Any:
        """
        Обработка шаблона с подстановкой переменных.
        
        Args:
            template_obj: Объект шаблона
            variables: Переменные для подстановки
            
        Returns:
            Обработанный объект
        """
        if isinstance(template_obj, str):
            # Обработка строки
            if template_obj.startswith("{{") and template_obj.endswith("}}"):
                var_name = template_obj[2:-2].strip()
                
                # Обработка диапазона значений (например, {{0.5-0.8}})
                if "-" in var_name and all(part.replace(".", "").isdigit() for part in var_name.split("-")):
                    min_val, max_val = map(float, var_name.split("-"))
                    return round(random.uniform(min_val, max_val), 2)
                
                # Обработка переменной
                return variables.get(var_name, template_obj)
            return template_obj
        elif isinstance(template_obj, list):
            # Обработка списка
            return [self._process_template(item, variables) for item in template_obj]
        elif isinstance(template_obj, dict):
            # Обработка словаря
            return {key: self._process_template(value, variables) for key, value in template_obj.items()}
        else:
            # Остальные типы данных
            return template_obj
    
    async def _execute_dream_step(self, step: Dict[str, Any], dream_session: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполнение шага сновидения.
        
        Args:
            step: Описание шага
            dream_session: Текущая сессия сновидения
            scenario: Сценарий сновидения
            
        Returns:
            Результаты выполнения шага
        """
        step_type = step["type"]
        step_description = step["description"]
        
        # Подготовка запроса к языковой модели
        prompt = self._prepare_step_prompt(step, dream_session, scenario)
        
        # Получение ответа от языковой модели
        llm_response = await self._get_llm_response(prompt)
        
        # Обработка ответа
        try:
            response_json = json.loads(llm_response)
        except:
            # Если не удалось разобрать JSON, пытаемся извлечь его из текста
            try:
                json_start = llm_response.find("{")
                json_end = llm_response.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = llm_response[json_start:json_end]
                    response_json = json.loads(json_str)
                else:
                    # Создание базовой структуры ответа
                    response_json = {
                        "step_type": step_type,
                        "description": "Не удалось получить структурированный ответ",
                        "content": llm_response
                    }
            except:
                # Если всё равно не удалось, создаем базовую структуру
                response_json = {
                    "step_type": step_type,
                    "description": "Не удалось получить структурированный ответ",
                    "content": llm_response
                }
        
        # Добавление метаданных шага
        response_json["step_type"] = step_type
        response_json["step_description"] = step_description
        response_json["timestamp"] = datetime.datetime.now().isoformat()
        
        return response_json
    
    def _prepare_step_prompt(self, step: Dict[str, Any], dream_session: Dict[str, Any], scenario: Dict[str, Any]) -> str:
        """
        Подготовка промпта для шага сновидения.
        
        Args:
            step: Описание шага
            dream_session: Текущая сессия сновидения
            scenario: Сценарий сновидения
            
        Returns:
            Промпт для языковой модели
        """
        # Базовая информация о сценарии
        prompt = f"""
# SASOK Dream Engine - Симуляция сценария

## Сценарий: {scenario['name']}
Описание: {scenario['description']}
Тип: {scenario['type']}
Сложность: {scenario['difficulty']}

## Текущий шаг: {step['type']} - {step['description']}
"""
        
        # Добавление информации о предыдущих шагах
        if dream_session["steps"]:
            prompt += "\n## Предыдущие шаги:\n"
            for prev_step in dream_session["steps"]:
                prompt += f"- {prev_step['type']}: {prev_step['description']}\n"
                
                # Добавление краткой информации о результате
                if "insights" in prev_step["result"]:
                    prompt += "  Инсайты:\n"
                    for insight in prev_step["result"]["insights"]:
                        prompt += f"  - {insight}\n"
        
        # Добавление текущих переменных
        prompt += "\n## Текущие переменные и состояние:\n```json\n"
        prompt += json.dumps(dream_session["variables"], indent=2, ensure_ascii=False)
        prompt += "\n```\n"
        
        # Инструкции в зависимости от типа шага
        prompt += "\n## Инструкции для текущего шага:\n"
        
        if step["type"] == "emotional_reaction":
            prompt += """
Ты симулируешь эмоциональную реакцию системы SASOK на ситуацию.
Рассмотри, как система должна эмоционально отреагировать на текущее состояние пользователя.
Учти конфликтующие эмоции, неоднозначность ситуации и цели системы.
"""
        elif step["type"] == "analysis":
            prompt += """
Ты анализируешь ситуацию, раскрывая причины, факторы и контекст.
Выяви скрытые мотивы, подсознательные элементы и возможные скрытые паттерны.
Используй аналитический подход, но учитывай эмоциональный контекст.
"""
        elif step["type"] == "resolution_attempt":
            prompt += """
Ты моделируешь попытку системы разрешить конфликт или проблему.
Предложи стратегию и тактику разрешения, учитывая эмоции и контекст.
Рассмотри, как система может помочь пользователю найти выход из ситуации.
"""
        elif step["type"] == "learning":
            prompt += """
Ты формулируешь уроки и инсайты, которые система SASOK получает из этой симуляции.
Выдели ключевые уроки, связанные с улучшением взаимодействия с пользователем.
Сформулируй конкретные действия, которые система может предпринять в будущем.
"""
        elif step["type"] == "dilemma_recognition":
            prompt += """
Ты моделируешь процесс осознания этической дилеммы системой SASOK.
Четко сформулируй, в чем состоит этическая дилемма и почему она является дилеммой.
Опиши внутренние противоречия, которые испытывает система при столкновении с этой дилеммой.
"""
        elif step["type"] == "values_analysis":
            prompt += """
Ты анализируешь конфликтующие ценности в этической дилемме.
Рассмотри, какие ценности вступают в противоречие и почему.
Оцени относительную важность этих ценностей в данном контексте.
"""
        elif "adaptation" in step["type"]:
            prompt += """
Ты симулируешь процесс адаптации системы SASOK к пользователю.
Опиши, как система распознает паттерны и настраивает свое поведение.
Какие конкретные изменения она вносит в свой подход к взаимодействию.
"""
        else:
            prompt += """
Смоделируй текущий шаг симуляции, учитывая контекст, предыдущие шаги и текущее состояние.
Будь креативным, но логичным в рамках сценария.
"""
        
        # Формат ответа
        prompt += """
## Формат ответа:
Предоставь ответ в формате JSON со следующей структурой:
```json
{
  "summary": "Краткое описание результата шага",
  "details": "Подробное описание процесса и выводов",
  "insights": ["Список ключевых инсайтов"],
  "outcomes": ["Список конкретных результатов или уроков"],
  "state_changes": {
    "ключ_состояния": "новое_значение"
  }
}