"""
Transparency Module for SASOK
Provides explanations for system decisions and manages user visibility configurations.
"""
from typing import Dict, Any

class TransparencyModule:
    def __init__(self, visibility_config: Dict[str, Any] = None):
        self.visibility_config = visibility_config or {}

    async def build_explanation(self, decision_data: Dict[str, Any], detail_level: str = "medium") -> Dict[str, Any]:
        explanation = {
            "decision": decision_data.get("decision"),
            "reasoning": decision_data.get("reasoning", "No reasoning provided."),
            "detail_level": detail_level,
            "visible_to": self.visibility_config.get("user", "all")
        }
        return explanation

    def set_visibility(self, user_id: str, level: str):
        self.visibility_config[user_id] = level

    def get_visibility(self, user_id: str) -> str:
        return self.visibility_config.get(user_id, "default")
