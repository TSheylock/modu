"""
Aesthetics Module for SASOK
Manages stylistic profiles, emotive templates, and poetic bias weights.
"""
import json
from typing import Dict, Any

class AestheticsModule:
    def __init__(self, config_path: str = None):
        self.style_profiles = {}
        self.emotive_templates = {}
        self.poetic_bias_weights = {}
        if config_path:
            self.load_config(config_path)

    def load_config(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.style_profiles = config.get('style_profiles', {})
        self.emotive_templates = config.get('emotive_templates', {})
        self.poetic_bias_weights = config.get('poetic_bias_weights', {})

    def get_template(self, category: str, emotion: str = "neutral") -> str:
        return self.emotive_templates.get(category, {}).get(emotion, "[No template found]")

    def set_poetic_bias(self, category: str, weight: float):
        self.poetic_bias_weights[category] = weight

    def get_style_profile(self, profile_name: str) -> Dict[str, Any]:
        return self.style_profiles.get(profile_name, {})

    def update_style_profile(self, profile_name: str, profile_data: Dict[str, Any]):
        self.style_profiles[profile_name] = profile_data

    def save_config(self, path: str):
        config = {
            'style_profiles': self.style_profiles,
            'emotive_templates': self.emotive_templates,
            'poetic_bias_weights': self.poetic_bias_weights
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
