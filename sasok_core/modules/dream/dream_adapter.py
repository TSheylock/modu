"""
Адаптер DreamModule для SASOK BaseModule.
Оборачивает ScenarioBank и DreamRunner в интерфейс BaseModule.
"""
import json
import asyncio
from typing import Dict, Any
from core.base_module import BaseModule


class DreamModule(BaseModule):
    """Модуль сновидений SASOK — автономное моделирование сценариев."""

    async def initialize(self):
        """Инициализация модуля сновидений."""
        self.logger.info("Инициализация модуля сновидений...")

        self.scenario_bank = None
        self.dream_runner = None
        self._dream_task = None

        self.state = {
            "active": False,
            "dreams_completed": 0,
            "last_dream": None,
            "insights_total": 0,
            "mode": "idle"  # idle | dreaming | processing
        }

        # Ленивая загрузка ScenarioBank
        try:
            from modules.dream.dream_module import ScenarioBank
            self.scenario_bank = ScenarioBank()
            self.logger.info(
                f"ScenarioBank загружен: {len(self.scenario_bank.scenarios)} сценариев"
            )
        except Exception as e:
            self.logger.warning(f"ScenarioBank недоступен: {e}")

        self.logger.info("Модуль сновидений инициализирован")

    async def activate(self):
        """Активация модуля сновидений."""
        if self.active:
            return

        self.logger.info("Активация модуля сновидений...")

        # Подписка на события, запускающие сновидения
        await self.subscribe("system.idle", self._on_system_idle)
        await self.subscribe("dream.trigger", self._on_dream_trigger)
        await self.subscribe("ethics.dilemma_detected", self._on_ethical_dilemma)

        self.active = True
        await self.update_state({"active": True})
        self.logger.info("Модуль сновидений активирован")

    async def deactivate(self):
        """Деактивация модуля сновидений."""
        if not self.active:
            return

        self.logger.info("Деактивация модуля сновидений...")

        # Остановка активного сновидения
        if self._dream_task and not self._dream_task.done():
            self._dream_task.cancel()
            try:
                await self._dream_task
            except asyncio.CancelledError:
                pass

        for subscription in self.subscriptions:
            await subscription.unsubscribe()
        self.subscriptions = []

        self.active = False
        await self.update_state({"active": False, "mode": "idle"})
        self.logger.info("Модуль сновидений деактивирован")

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка запроса на сновидение."""
        if not self.active:
            return {"error": "Module inactive"}

        if not self.scenario_bank:
            return {"error": "ScenarioBank not available", "success": False}

        action = data.get("action", "run_dream")

        if action == "run_dream":
            return await self._run_dream(
                scenario_id=data.get("scenario_id"),
                scenario_type=data.get("scenario_type")
            )
        elif action == "list_scenarios":
            scenarios = self.scenario_bank.get_all_scenarios()
            return {
                "scenarios": [
                    {"id": s["id"], "name": s["name"], "type": s["type"]}
                    for s in scenarios
                ]
            }
        elif action == "get_status":
            return {"state": self.state}
        else:
            return {"error": f"Unknown action: {action}"}

    async def _run_dream(
        self,
        scenario_id: str = None,
        scenario_type: str = None
    ) -> Dict[str, Any]:
        """Запуск сессии сновидения (упрощённая версия без LLM)."""
        import random
        import datetime

        # Выбор сценария
        if scenario_id:
            scenario = self.scenario_bank.get_scenario(scenario_id)
        else:
            scenario = self.scenario_bank.get_random_scenario(scenario_type)

        if not scenario:
            return {"error": "No scenario found", "success": False}

        await self.update_state({"mode": "dreaming"})
        self.logger.info(f"Сновидение: {scenario['name']} (тип: {scenario['type']})")

        # Симуляция шагов сценария
        steps_results = []
        template = scenario.get("template", {})
        variables = template.get("variables", {})

        # Генерируем начальное состояние
        initial_vars = {}
        for var_name, var_values in variables.items():
            if isinstance(var_values, list):
                initial_vars[var_name] = random.choice(var_values)

        for step in template.get("steps", []):
            step_result = {
                "type": step["type"],
                "description": step["description"],
                "timestamp": datetime.datetime.now().isoformat(),
                "insights": [],
                "outcomes": []
            }

            # Генерация инсайтов для шагов типа learning
            if step["type"] == "learning":
                step_result["insights"] = [
                    f"Обнаружен паттерн в сценарии '{scenario['name']}'",
                    f"Переменные: {json.dumps(initial_vars, ensure_ascii=False)}"
                ]
                step_result["outcomes"] = [
                    f"Улучшено понимание типа '{scenario['type']}'"
                ]

            steps_results.append(step_result)

        # Подсчёт инсайтов
        total_insights = sum(len(s.get("insights", [])) for s in steps_results)

        self.state["dreams_completed"] += 1
        self.state["last_dream"] = scenario["name"]
        self.state["insights_total"] += total_insights
        await self.update_state({"mode": "idle"})

        # Публикация результатов
        result = {
            "success": True,
            "scenario": scenario["name"],
            "type": scenario["type"],
            "steps_completed": len(steps_results),
            "insights_generated": total_insights,
            "initial_variables": initial_vars
        }

        await self.publish(
            "dream.completed",
            json.dumps(result).encode("utf-8")
        )

        return result

    async def _on_system_idle(self, msg):
        """Запуск сновидения при простое системы."""
        if self.state.get("mode") == "dreaming":
            return
        self._dream_task = asyncio.create_task(
            self._run_dream(scenario_type="behavioral")
        )

    async def _on_dream_trigger(self, msg):
        """Явный запрос на запуск сновидения."""
        data = json.loads(msg.data.decode())
        await self.process(data)

    async def _on_ethical_dilemma(self, msg):
        """Запуск этического сновидения при обнаружении дилеммы."""
        if self.state.get("mode") == "dreaming":
            return
        self._dream_task = asyncio.create_task(
            self._run_dream(scenario_type="ethical")
        )
