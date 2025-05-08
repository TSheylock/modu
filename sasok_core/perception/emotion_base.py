"""
Базовый класс для анализа эмоций в SASOK

Определяет интерфейсы для всех анализаторов эмоций, обеспечивая
единый архетипический подход к обработке эмоциональных данных.
"""
import os
import json
import time
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SASOK.Perception.Base")

class EmotionAnalyzer(ABC):
    """
    Абстрактный базовый класс для всех анализаторов эмоций
    Реализует метакогнитивные механизмы SASOK
    """
    
    def __init__(self, source_type: str):
        """
        Инициализация базового анализатора эмоций
        
        Args:
            source_type: Тип источника эмоций (текст, аудио, видео, др.)
        """
        self.source_type = source_type
        self.model_path = None
        self.config = {}
        self.tags = []  # Метакогнитивные теги
        self.created_at = datetime.now().isoformat()
        self.last_analysis = None
        self.confidence_threshold = 0.7  # Порог для меток SASOK_DOUBT
        self.archetype_patterns = {}  # Архетипические паттерны эмоций
        self.emotion_history = []  # История эмоциональных состояний
        self.calibration_data = {}  # Данные для калибровки по пользователю
        self.meta_state = {  # Метакогнитивное состояние
            "doubt_level": 0.0,
            "adaptation_level": 0.0,
            "self_correction_count": 0,
            "last_introspection": None
        }
        
        logger.info(f"Инициализирован базовый анализатор эмоций: {source_type}")
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Асинхронная инициализация анализатора
        
        Returns:
            bool: Успешность инициализации
        """
        pass
    
    @abstractmethod
    async def analyze(self, data: Any) -> Dict:
        """
        Анализ эмоций из данных
        
        Args:
            data: Данные для анализа (текст, аудио, изображение)
            
        Returns:
            Dict: Словарь с проанализированными эмоциями
        """
        pass
    
    async def check_confidence(self, emotions: Dict) -> Dict:
        """
        Проверка уверенности в определении эмоций
        Добавляет метакогнитивный флаг SASOK_DOUBT при низкой уверенности
        
        Args:
            emotions: Словарь с эмоциями и их значениями
            
        Returns:
            Dict: Обновленный словарь с возможным флагом SASOK_DOUBT
        """
        # Находим максимальное значение эмоции
        if not emotions:
            emotions = {"SASOK_DOUBT": True}
            self.meta_state["doubt_level"] += 0.1
            return emotions
        
        max_emotion = max(emotions.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
        
        # Если максимальное значение ниже порога, добавляем флаг
        if isinstance(max_emotion[1], (int, float)) and max_emotion[1] < self.confidence_threshold:
            emotions["SASOK_DOUBT"] = True
            self.meta_state["doubt_level"] += 0.05
        else:
            self.meta_state["doubt_level"] = max(0.0, self.meta_state["doubt_level"] - 0.01)
        
        return emotions
        
    async def apply_archetype_patterns(self, emotions: Dict) -> Dict:
        """
        Применение архетипических паттернов к эмоциям
        
        Args:
            emotions: Словарь с эмоциями
            
        Returns:
            Dict: Обновленный словарь с архетипическими метками
        """
        # Базовые архетипы в эмоциональном контексте
        archetypes = {
            "hero": ["радость", "уверенность", "решительность"],
            "shadow": ["страх", "злость", "ненависть"],
            "anima": ["любовь", "эмпатия", "нежность"],
            "animus": ["уважение", "расчетливость", "логичность"],
            "trickster": ["удивление", "игривость", "хитрость"],
            "sage": ["спокойствие", "мудрость", "созерцательность"],
            "caregiver": ["забота", "беспокойство", "сострадание"]
        }
        
        # Проверка доминирующих эмоций и соотнесение с архетипами
        dominant_archetype = None
        max_score = 0.0
        
        for archetype, emotion_list in archetypes.items():
            score = sum(emotions.get(e, 0) for e in emotion_list)
            if score > max_score:
                max_score = score
                dominant_archetype = archetype
        
        if dominant_archetype and max_score > 0.3:
            emotions["archetype"] = dominant_archetype
            
        return emotions
    
    async def format_result(self, emotions: Dict, metadata: Dict = None) -> Dict:
        """
        Форматирование результата анализа эмоций
        
        Args:
            emotions: Словарь с эмоциями
            metadata: Дополнительные метаданные
            
        Returns:
            Dict: Отформатированный результат
        """
        timestamp = datetime.now().isoformat()
        
        result = {
            "source": self.source_type,
            "emotions": emotions,
            "timestamp": timestamp,
            "tags": self.tags.copy()
        }
        
        if metadata:
            result["metadata"] = metadata
        
        self.last_analysis = result
        return result
