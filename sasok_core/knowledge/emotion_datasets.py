"""
Модуль для интеграции эмоциональных датасетов в семантическую сеть SASOK

Загружает и обрабатывает данные из EmpatheticDialogues, GoEmotions и других
эмоциональных датасетов, интегрируя их в Neo4j граф и предоставляя 
функциональность для обучения и тонкой настройки моделей эмоций.
"""
import os
import logging
import json
import asyncio
import time
import csv
import hashlib
import random
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import zipfile
import requests
import io

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments
)
from sklearn.model_selection import train_test_split

from sasok_core.knowledge.knowledge_base import KnowledgeBase
from sasok_core.utils.nats_client import NatsClient

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SASOK.EmotionDatasets")

# Путь к директории с данными
DATA_DIR = Path(os.getenv("DATA_DIR", "/home/sasok/Рабочий стол/blackboxai-1745739396945/data"))
EMOTION_DATASETS_DIR = DATA_DIR / "emotion_datasets"
EMOTION_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# Поддиректории для разных датасетов
EMPATHETIC_DIALOGUES_DIR = EMOTION_DATASETS_DIR / "empathetic_dialogues"
GO_EMOTIONS_DIR = EMOTION_DATASETS_DIR / "go_emotions"
EMPATHETIC_DIALOGUES_DIR.mkdir(exist_ok=True)
GO_EMOTIONS_DIR.mkdir(exist_ok=True)

# URLs для загрузки датасетов
EMPATHETIC_DIALOGUES_URL = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
GO_EMOTIONS_URL = "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv"

# Список эмоций из EmpatheticDialogues
EMPATHETIC_EMOTIONS = [
    "afraid", "angry", "annoyed", "anticipating", "anxious", "apprehensive", "ashamed", 
    "caring", "confident", "content", "devastated", "disappointed", "disgusted", 
    "embarrassed", "excited", "faithful", "furious", "grateful", "guilty", "hopeful", 
    "impressed", "jealous", "joyful", "lonely", "nostalgic", "prepared", "proud", 
    "sad", "sentimental", "surprised", "terrified", "trusting"
]

# Список эмоций из GoEmotions (основные категории)
GO_EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

class EmotionDatasetsProcessor(KnowledgeBase):
    """
    Класс для обработки и интеграции эмоциональных датасетов
    """
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, lang: str = "ru"):
        """
        Инициализация процессора эмоциональных датасетов
        
        Args:
            neo4j_uri: URI для подключения к Neo4j
            neo4j_user: Имя пользователя Neo4j
            neo4j_password: Пароль пользователя Neo4j
            lang: Основной язык (для перевода и обработки)
        """
        super().__init__(neo4j_uri, neo4j_user, neo4j_password)
        self.lang = lang
        self.nats_client = None
        self.datasets_cache = {}
        
        # Создание индексов в Neo4j
        self.create_index("EmotionExample", "id")
        self.create_index("EmotionExample", "emotion")
        self.create_index("Emotion", "name")
    
    async def initialize(self):
        """
        Асинхронная инициализация, включая подключение к NATS
        """
        # Подключение к NATS
        nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
        self.nats_client = await NatsClient.get_instance(nats_url)
        logger.info("EmotionDatasetsProcessor инициализирован и подключен к NATS")

    def download_empathetic_dialogues(self) -> bool:
        """
        Загрузка датасета EmpatheticDialogues
        
        Returns:
            True если загрузка успешна, иначе False
        """
        # Проверка наличия файла
        target_file = EMPATHETIC_DIALOGUES_DIR / "empatheticdialogues.tar.gz"
        if target_file.exists():
            logger.info(f"Файл {target_file} уже существует, пропускаем загрузку")
            return True
        
        # Загрузка файла
        try:
            logger.info(f"Загрузка EmpatheticDialogues из {EMPATHETIC_DIALOGUES_URL}")
            response = requests.get(EMPATHETIC_DIALOGUES_URL, stream=True)
            if response.status_code == 200:
                with open(target_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Файл успешно загружен: {target_file}")
                
                # Распаковка архива
                import tarfile
                with tarfile.open(target_file, "r:gz") as tar:
                    tar.extractall(path=EMPATHETIC_DIALOGUES_DIR)
                logger.info(f"Архив распакован в {EMPATHETIC_DIALOGUES_DIR}")
                
                return True
            else:
                logger.error(f"Ошибка загрузки EmpatheticDialogues: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Ошибка при загрузке EmpatheticDialogues: {e}")
            return False
    
    def download_go_emotions(self) -> bool:
        """
        Загрузка датасета GoEmotions
        
        Returns:
            True если загрузка успешна, иначе False
        """
        # Проверка наличия файла
        target_file = GO_EMOTIONS_DIR / "goemotions_1.csv"
        if target_file.exists():
            logger.info(f"Файл {target_file} уже существует, пропускаем загрузку")
            return True
        
        # Загрузка файла
        try:
            logger.info(f"Загрузка GoEmotions из {GO_EMOTIONS_URL}")
            response = requests.get(GO_EMOTIONS_URL)
            if response.status_code == 200:
                with open(target_file, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Файл успешно загружен: {target_file}")
                return True
            else:
                logger.error(f"Ошибка загрузки GoEmotions: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Ошибка при загрузке GoEmotions: {e}")
            return False
    
    def process_empathetic_dialogues(self) -> Dict[str, List[Dict]]:
        """
        Обработка данных EmpatheticDialogues
        
        Returns:
            Словарь с примерами по эмоциям
        """
        # Проверка кэша
        cache_key = "empathetic_dialogues_processed"
        if cache_key in self.datasets_cache:
            return self.datasets_cache[cache_key]
        
        # Проверка наличия данных
        data_file = EMPATHETIC_DIALOGUES_DIR / "empatheticdialogues" / "train.csv"
        if not data_file.exists():
            logger.warning(f"Файл данных EmpatheticDialogues не найден: {data_file}")
            if not self.download_empathetic_dialogues():
                return {}
        
        # Чтение и обработка данных
        try:
            # Структура для хранения примеров по эмоциям
            emotion_examples = {emotion: [] for emotion in EMPATHETIC_EMOTIONS}
            
            # Чтение CSV файла
            with open(data_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Пропуск заголовка
                
                for row in reader:
                    try:
                        # Извлечение данных
                        conv_id, utterance_idx, prompt, context, speaker, utterance, emotion, _ = row
                        
                        # Добавление примера в соответствующую эмоцию
                        if emotion in emotion_examples:
                            example = {
                                "id": f"{conv_id}_{utterance_idx}",
                                "emotion": emotion,
                                "context": context,
                                "utterance": utterance,
                                "prompt": prompt
                            }
                            emotion_examples[emotion].append(example)
                    except Exception as e:
                        logger.warning(f"Ошибка обработки строки: {e}")
                        continue
            
            # Кэширование результата
            self.datasets_cache[cache_key] = emotion_examples
            
            logger.info(f"Обработано примеров EmpatheticDialogues: {sum(len(examples) for examples in emotion_examples.values())}")
            return emotion_examples
        except Exception as e:
            logger.error(f"Ошибка при обработке EmpatheticDialogues: {e}")
            return {}
    
    def process_go_emotions(self) -> Dict[str, List[Dict]]:
        """
        Обработка данных GoEmotions
        
        Returns:
            Словарь с примерами по эмоциям
        """
        # Проверка кэша
        cache_key = "go_emotions_processed"
        if cache_key in self.datasets_cache:
            return self.datasets_cache[cache_key]
        
        # Проверка наличия данных
        data_file = GO_EMOTIONS_DIR / "goemotions_1.csv"
        if not data_file.exists():
            logger.warning(f"Файл данных GoEmotions не найден: {data_file}")
            if not self.download_go_emotions():
                return {}
        
        # Чтение и обработка данных
        try:
            # Структура для хранения примеров по эмоциям
            emotion_examples = {emotion: [] for emotion in GO_EMOTIONS}
            
            # Чтение CSV файла с помощью pandas
            df = pd.read_csv(data_file)
            
            # Обработка каждой строки
            for idx, row in df.iterrows():
                try:
                    # Определение эмоций в строке
                    emotions_cols = GO_EMOTIONS
                    active_emotions = [emotion for emotion in emotions_cols if row.get(emotion, 0) == 1]
                    
                    # Если есть активные эмоции, добавляем пример
                    if active_emotions:
                        text = row.get('text', '')
                        for emotion in active_emotions:
                            example = {
                                "id": f"go_{idx}_{emotion}",
                                "emotion": emotion,
                                "text": text,
                                "source": "reddit"
                            }
                            emotion_examples[emotion].append(example)
                except Exception as e:
                    logger.warning(f"Ошибка обработки строки {idx}: {e}")
                    continue
            
            # Кэширование результата
            self.datasets_cache[cache_key] = emotion_examples
            
            logger.info(f"Обработано примеров GoEmotions: {sum(len(examples) for examples in emotion_examples.values())}")
            return emotion_examples
        except Exception as e:
            logger.error(f"Ошибка при обработке GoEmotions: {e}")
            return {}
    
    def import_emotions_to_neo4j(self, emotions: List[str]) -> List[Dict]:
        """
        Импорт эмоций в Neo4j
        
        Args:
            emotions: Список эмоций для импорта
            
        Returns:
            Список созданных узлов
        """
        created_nodes = []
        
        for emotion in emotions:
            # Создание узла эмоции
            emotion_node = self.create_node(
                "Emotion",
                id=hashlib.md5(emotion.encode()).hexdigest(),
                name=emotion,
                category="basic" if emotion in ["joy", "sadness", "anger", "fear", "surprise", "disgust"] else "complex",
                created_at=time.time()
            )
            
            if emotion_node:
                created_nodes.append({
                    "id": emotion_node["id"],
                    "name": emotion_node["name"]
                })
                logger.info(f"Эмоция импортирована: {emotion}")
            
        return created_nodes
    
    def import_emotion_examples_to_neo4j(self, examples: Dict[str, List[Dict]]) -> int:
        """
        Импорт примеров эмоций в Neo4j
        
        Args:
            examples: Словарь с примерами по эмоциям
            
        Returns:
            Количество импортированных примеров
        """
        imported_count = 0
        
        for emotion, emotion_examples in examples.items():
            # Получение узла эмоции
            emotion_node = None
            query = "MATCH (e:Emotion {name: $name}) RETURN e"
            result = self.run_query(query, name=emotion)
            
            if result:
                emotion_node = result[0]["e"]
            else:
                # Создание узла эмоции, если не существует
                emotion_node = self.create_node(
                    "Emotion",
                    id=hashlib.md5(emotion.encode()).hexdigest(),
                    name=emotion,
                    category="basic" if emotion in ["joy", "sadness", "anger", "fear", "surprise", "disgust"] else "complex",
                    created_at=time.time()
                )
            
            if not emotion_node:
                logger.warning(f"Не удалось получить/создать узел эмоции: {emotion}")
                continue
            
            # Импорт примеров
            for example in emotion_examples:
                # Проверка наличия примера
                example_id = example.get("id", hashlib.md5(str(example).encode()).hexdigest())
                query = "MATCH (e:EmotionExample {id: $id}) RETURN e"
                result = self.run_query(query, id=example_id)
                
                if result:
                    # Пример уже существует, пропускаем
                    continue
                
                # Создание узла примера
                example_node = self.create_node(
                    "EmotionExample",
                    id=example_id,
                    emotion=emotion,
                    text=example.get("text", example.get("utterance", "")),
                    context=example.get("context", ""),
                    source=example.get("source", "unknown"),
                    created_at=time.time()
                )
                
                if example_node:
                    # Создание связи с эмоцией
                    self.create_relationship(
                        example_node,
                        emotion_node,
                        "HAS_EMOTION",
                        confidence=1.0
                    )
                    
                    imported_count += 1
                    
                    if imported_count % 100 == 0:
                        logger.info(f"Импортировано примеров: {imported_count}")
        
        logger.info(f"Всего импортировано примеров: {imported_count}")
        return imported_count
    
    async def import_all_datasets(self):
        """
        Импорт всех доступных датасетов
        """
        # Загрузка и обработка EmpatheticDialogues
        logger.info("Импорт датасета EmpatheticDialogues")
        if self.download_empathetic_dialogues():
            ed_examples = self.process_empathetic_dialogues()
            # Импорт эмоций
            self.import_emotions_to_neo4j(EMPATHETIC_EMOTIONS)
            # Импорт примеров (выборка для тестирования)
            samples = {emotion: examples[:100] for emotion, examples in ed_examples.items()}
            self.import_emotion_examples_to_neo4j(samples)
            
            # Публикация события
            if self.nats_client and self.nats_client.connected:
                event = {
                    "type": "dataset_import_completed",
                    "dataset": "EmpatheticDialogues",
                    "emotions_count": len(EMPATHETIC_EMOTIONS),
                    "examples_count": sum(len(examples) for examples in samples.values()),
                    "timestamp": time.time()
                }
                
                await self.nats_client.publish(
                    "sasok.knowledge.datasets.import",
                    json.dumps(event).encode()
                )
        
        # Загрузка и обработка GoEmotions
        logger.info("Импорт датасета GoEmotions")
        if self.download_go_emotions():
            ge_examples = self.process_go_emotions()
            # Импорт эмоций
            self.import_emotions_to_neo4j(GO_EMOTIONS)
            # Импорт примеров (выборка для тестирования)
            samples = {emotion: examples[:100] for emotion, examples in ge_examples.items()}
            self.import_emotion_examples_to_neo4j(samples)
            
            # Публикация события
            if self.nats_client and self.nats_client.connected:
                event = {
                    "type": "dataset_import_completed",
                    "dataset": "GoEmotions",
                    "emotions_count": len(GO_EMOTIONS),
                    "examples_count": sum(len(examples) for examples in samples.values()),
                    "timestamp": time.time()
                }
                
                await self.nats_client.publish(
                    "sasok.knowledge.datasets.import",
                    json.dumps(event).encode()
                )
    
    class EmotionDataset(Dataset):
        """
        Класс для создания датасета эмоций для обучения моделей
        """
        def __init__(self, texts, emotions, tokenizer, max_length=128):
            """
            Инициализация датасета
            
            Args:
                texts: Список текстов
                emotions: Список меток эмоций
                tokenizer: Токенизатор трансформера
                max_length: Максимальная длина токенизированной последовательности
            """
            self.texts = texts
            self.emotions = emotions
            self.tokenizer = tokenizer
            self.max_length = max_length
            
            # Создание словаря соответствия меток эмоций и индексов
            unique_emotions = sorted(list(set(emotions)))
            self.emotion_to_id = {emotion: i for i, emotion in enumerate(unique_emotions)}
            self.id_to_emotion = {i: emotion for emotion, i in self.emotion_to_id.items()}
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = self.texts[idx]
            emotion = self.emotions[idx]
            emotion_id = self.emotion_to_id[emotion]
            
            # Токенизация текста
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Удаление дополнительного измерения батча
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': torch.tensor(emotion_id, dtype=torch.long)
            }
    
    async def prepare_training_data(self, dataset_name: str = "combined") -> Tuple[List, List]:
        """
        Подготовка данных для обучения моделей
        
        Args:
            dataset_name: Имя датасета для подготовки ("empathetic_dialogues", "go_emotions", "combined")
            
        Returns:
            Кортеж (тексты, метки эмоций)
        """
        texts = []
        emotions = []
        
        if dataset_name in ["empathetic_dialogues", "combined"]:
            # Загрузка и обработка EmpatheticDialogues
            ed_examples = self.process_empathetic_dialogues()
            for emotion, examples in ed_examples.items():
                for example in examples:
                    text = example.get("utterance", "")
                    if text:
                        texts.append(text)
                        emotions.append(emotion)
        
        if dataset_name in ["go_emotions", "combined"]:
            # Загрузка и обработка GoEmotions
            ge_examples = self.process_go_emotions()
            for emotion, examples in ge_examples.items():
                for example in examples:
                    text = example.get("text", "")
                    if text:
                        texts.append(text)
                        emotions.append(emotion)
        
        logger.info(f"Подготовлено {len(texts)} примеров для обучения")
        
        # Публикация события
        if self.nats_client and self.nats_client.connected:
            event = {
                "type": "training_data_prepared",
                "dataset": dataset_name,
                "samples_count": len(texts),
                "emotions_count": len(set(emotions)),
                "timestamp": time.time()
            }
            
            await self.nats_client.publish(
                "sasok.knowledge.datasets.training",
                json.dumps(event).encode()
            )
        
        return texts, emotions
    
    async def fine_tune_model(self, model_name: str = "distilbert-base-uncased", dataset_name: str = "combined", output_dir: str = None):
        """
        Тонкая настройка модели на эмоциональном датасете
        
        Args:
            model_name: Название предобученной модели
            dataset_name: Имя датасета для обучения
            output_dir: Директория для сохранения обученной модели
        """
        if output_dir is None:
            output_dir = str(MODELS_DIR / f"emotion_{dataset_name}_{int(time.time())}")
        
        # Подготовка данных
        texts, emotions = await self.prepare_training_data(dataset_name)
        
        if not texts:
            logger.error("Нет данных для обучения")
            return
        
        # Разделение на тренировочный и тестовый наборы
        train_texts, val_texts, train_emotions, val_emotions = train_test_split(
            texts, emotions, test_size=0.1, random_state=42
        )
        
        # Загрузка токенизатора и модели
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            unique_emotions = sorted(list(set(emotions)))
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=len(unique_emotions)
            )
            
            # Создание датасетов
            train_dataset = self.EmotionDataset(train_texts, train_emotions, tokenizer)
            val_dataset = self.EmotionDataset(val_texts, val_emotions, tokenizer)
            
            # Настройка обучения
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=3,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=f"{output_dir}/logs",
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
            )
            
            # Создание тренера
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset
            )
            
            # Обучение модели
            logger.info(f"Начало обучения модели {model_name} на датасете {dataset_name}")
            trainer.train()
            
            # Сохранение модели
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Сохранение маппинга эмоций
            emotion_mapping = {
                "emotion_to_id": train_dataset.emotion_to_id,
                "id_to_emotion": train_dataset.id_to_emotion
            }
            with open(f"{output_dir}/emotion_mapping.json", "w") as f:
                json.dump(emotion_mapping, f)
            
            logger.info(f"Модель успешно обучена и сохранена в {output_dir}")
            
            # Публикация события
            if self.nats_client and self.nats_client.connected:
                event = {
                    "type": "model_training_completed",
                    "model_name": model_name,
                    "dataset": dataset_name,
                    "output_dir": output_dir,
                    "emotions_count": len(unique_emotions),
                    "train_samples": len(train_texts),
                    "val_samples": len(val_texts),
                    "timestamp": time.time()
                }
                
                await self.nats_client.publish(
                    "sasok.knowledge.datasets.model_training",
                    json.dumps(event).encode()
                )
        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {e}")
            
            # Публикация события об ошибке
            if self.nats_client and self.nats_client.connected:
                event = {
                    "type": "model_training_error",
                    "model_name": model_name,
                    "dataset": dataset_name,
                    "error": str(e),
                    "timestamp": time.time()
                }
                
                await self.nats_client.publish(
                    "sasok.knowledge.datasets.model_training",
                    json.dumps(event).encode()
                )

# Пример использования
async def main():
    # Получение настроек из переменных окружения
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    
    # Создание процессора эмоциональных датасетов
    processor = EmotionDatasetsProcessor(neo4j_uri, neo4j_user, neo4j_password)
    await processor.initialize()
    
    # Импорт датасетов
    await processor.import_all_datasets()
    
    # Подготовка к обучению модели
    # await processor.fine_tune_model(
    #     model_name="distilbert-base-uncased",
    #     dataset_name="combined"
    # )
    
    # Закрытие соединений
    processor.close()

if __name__ == "__main__":
    asyncio.run(main())
