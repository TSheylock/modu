"""
Emotion Analysis Module for SASOK
Supports multimodal emotion analysis: text, audio, and video in real-time.
Integration with MediaPipe, DeepFace for video, Transformers for text/audio.
Publishes events to NATS event bus for further processing.
"""
from typing import Dict, Any, Optional, List, Union
import os
import asyncio
import json
import numpy as np
import cv2
import torch
from transformers import pipeline
import mediapipe as mp
from deepface import DeepFace
import nats
import base64
import logging
from pathlib import Path
import aiofiles

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EmotionAnalysis")

# Эмоциональный словарь для унификации выходов разных моделей
EMOTION_MAP = {
    # Map from DeepFace emotions
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "joy",
    "sad": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
    # Map from GoEmotions
    "admiration": "admiration",
    "amusement": "joy",
    "disappointment": "sadness",
    "embarrassment": "embarrassment",
    "excitement": "joy",
    "gratitude": "gratitude",
    "love": "love",
    "optimism": "joy",
    "remorse": "sadness",
    "POSITIVE": "positive",
    "NEGATIVE": "negative",
    "NEUTRAL": "neutral"
}

# Text emotion analysis using DistilBERT (or other Transformer)
class TextEmotionAnalyzer:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or "bhadresh-savani/distilbert-base-uncased-emotion"
        self.pipeline = None
        self.emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        
    def load_model(self):
        if self.pipeline is None:
            self.pipeline = pipeline("text-classification", model=self.model_name)
            logger.info(f"Loaded text emotion model: {self.model_name}")

    async def analyze(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {"emotion": "neutral", "score": 1.0}
            
        self.load_model()  # Load on demand
            
        try:
            result = self.pipeline(text)
            emotion = result[0]["label"].lower()
            
            # Маппинг к унифицированной эмоциональной таксономии
            mapped_emotion = EMOTION_MAP.get(emotion, emotion)
            
            return {
                "emotion": mapped_emotion,
                "score": float(result[0]["score"]),
                "raw_emotion": emotion,
                "timestamp": asyncio.get_event_loop().time()
            }
        except Exception as e:
            logger.error(f"Error in text emotion analysis: {e}")
            return {"emotion": "neutral", "score": 0.5, "error": str(e)}

# Audio emotion analysis using wav2vec (or similar)
class AudioEmotionAnalyzer:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        self.pipeline = None
        self.sample_rate = 16000
        
    def load_model(self):
        if self.pipeline is None:
            self.pipeline = pipeline("audio-classification", model=self.model_name)
            logger.info(f"Loaded audio emotion model: {self.model_name}")

    async def analyze(self, audio_data: Union[str, np.ndarray]) -> Dict[str, Any]:
        self.load_model()  # Load on demand
            
        try:
            # Если аудио передано как путь к файлу
            if isinstance(audio_data, str):
                result = self.pipeline(audio_data)
            # Если аудио передано как numpy array
            elif isinstance(audio_data, np.ndarray):
                result = self.pipeline({"array": audio_data, "sampling_rate": self.sample_rate})
            else:
                return {"emotion": "neutral", "score": 0.0, "error": "Unsupported audio format"}
                
            emotion = result[0]["label"].lower()
            
            # Маппинг к унифицированной эмоциональной таксономии
            mapped_emotion = EMOTION_MAP.get(emotion, emotion)
            
            return {
                "emotion": mapped_emotion,
                "score": float(result[0]["score"]),
                "raw_emotion": emotion,
                "timestamp": asyncio.get_event_loop().time()
            }
        except Exception as e:
            logger.error(f"Error in audio emotion analysis: {e}")
            return {"emotion": "neutral", "score": 0.5, "error": str(e)}

# Video emotion analysis using MediaPipe and DeepFace
class VideoEmotionAnalyzer:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = None
        self.temp_dir = Path("/tmp/sasok_emotion_frames")
        self.temp_dir.mkdir(exist_ok=True)
        
    def load_models(self):
        if self.face_detection is None:
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,  # Модель для более дальних расстояний
                min_detection_confidence=0.5
            )
            logger.info("Loaded MediaPipe Face Detection model")

    async def analyze_frame(self, frame_data: Union[str, np.ndarray]) -> Dict[str, Any]:
        self.load_models()
        
        try:
            # Получаем кадр: из пути или напрямую как numpy array
            if isinstance(frame_data, str):
                # Если передан путь к файлу
                frame = cv2.imread(frame_data)
            elif isinstance(frame_data, np.ndarray):
                # Если передан numpy array
                frame = frame_data
            else:
                return {"emotion": "neutral", "score": 0.0, "error": "Unsupported frame format"}
            
            if frame is None or frame.size == 0:
                return {"emotion": "neutral", "score": 0.0, "error": "Empty frame"}
            
            # Преобразование в RGB для MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Детекция лица с MediaPipe
            results = self.face_detection.process(rgb_frame)
            
            if not results.detections:
                return {"emotion": "neutral", "score": 0.0, "error": "No face detected"}
            
            # Временный файл для сохранения кадра (необходимо для DeepFace)
            temp_file = self.temp_dir / f"frame_{asyncio.get_event_loop().time()}.jpg"
            cv2.imwrite(str(temp_file), frame)
            
            # Анализ эмоций с DeepFace
            try:
                df_results = DeepFace.analyze(
                    img_path=str(temp_file),
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                
                if isinstance(df_results, list):
                    df_results = df_results[0]  # Берем первое лицо
                
                # Получаем основную эмоцию
                emotion = df_results['dominant_emotion'].lower()
                
                # Удаляем временный файл
                os.remove(temp_file)
                
                # Маппинг к унифицированной эмоциональной таксономии
                mapped_emotion = EMOTION_MAP.get(emotion, emotion)
                
                return {
                    "emotion": mapped_emotion,
                    "score": float(df_results['emotion'][emotion] / 100),
                    "raw_emotion": emotion,
                    "emotions_raw": {k: v/100 for k, v in df_results['emotion'].items()},
                    "face_detected": True,
                    "timestamp": asyncio.get_event_loop().time()
                }
            except Exception as e:
                logger.error(f"DeepFace error: {e}")
                return {
                    "emotion": "neutral", 
                    "score": 0.5, 
                    "error": f"DeepFace error: {e}",
                    "face_detected": True
                }
                
        except Exception as e:
            logger.error(f"Error in video emotion analysis: {e}")
            return {"emotion": "neutral", "score": 0.0, "error": str(e)}
    
    async def analyze(self, video_data: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Анализ видео (или отдельного кадра)
        """
        return await self.analyze_frame(video_data)

# NATS client для публикации событий в event bus
class EmotionEventPublisher:
    def __init__(self, nats_url: str = "nats://localhost:4222"):
        self.nats_url = nats_url
        self.client = None
        
    async def connect(self):
        if self.client is None or not self.client.is_connected:
            try:
                self.client = await nats.connect(self.nats_url)
                logger.info(f"Connected to NATS server at {self.nats_url}")
            except Exception as e:
                logger.error(f"Error connecting to NATS: {e}")
                raise
    
    async def publish_emotion(self, subject: str, emotion_data: Dict[str, Any]):
        """
        Публикует данные эмоций в NATS
        """
        try:
            await self.connect()
            payload = json.dumps(emotion_data).encode()
            await self.client.publish(subject, payload)
            logger.debug(f"Published emotion data to {subject}")
        except Exception as e:
            logger.error(f"Error publishing to NATS: {e}")
            
    async def close(self):
        if self.client and self.client.is_connected:
            await self.client.close()
            logger.info("Closed NATS connection")

# Класс для загрузки и управления эмоциональными датасетами
class EmotionalDatasetLoader:
    def __init__(self, datasets_dir: str = None):
        self.datasets_dir = datasets_dir or os.path.join(os.getcwd(), "datasets")
        self.loaded_datasets = {}
        
    async def load_empathetic_dialogues(self):
        """
        Загрузить EmpatheticDialogues dataset
        """
        try:
            from datasets import load_dataset
            
            dataset = load_dataset("empathetic_dialogues")
            self.loaded_datasets["empathetic_dialogues"] = dataset
            logger.info("Loaded EmpatheticDialogues dataset")
            return dataset
        except Exception as e:
            logger.error(f"Error loading EmpatheticDialogues: {e}")
            return None
    
    async def load_go_emotions(self):
        """
        Загрузить GoEmotions dataset
        """
        try:
            from datasets import load_dataset
            
            dataset = load_dataset("go_emotions")
            self.loaded_datasets["go_emotions"] = dataset
            logger.info("Loaded GoEmotions dataset")
            return dataset
        except Exception as e:
            logger.error(f"Error loading GoEmotions: {e}")
            return None
            
    async def get_examples_by_emotion(self, emotion: str, dataset_name: str = "empathetic_dialogues", limit: int = 5):
        """
        Получить примеры для конкретной эмоции
        """
        if dataset_name not in self.loaded_datasets:
            if dataset_name == "empathetic_dialogues":
                await self.load_empathetic_dialogues()
            elif dataset_name == "go_emotions":
                await self.load_go_emotions()
                
        dataset = self.loaded_datasets.get(dataset_name)
        if not dataset:
            return []
            
        examples = []
        if dataset_name == "empathetic_dialogues":
            # Ищем в EmpatheticDialogues
            for item in dataset["train"]:
                if item["emotion"].lower() == emotion.lower():
                    examples.append(item["utterance"])
                    if len(examples) >= limit:
                        break
        elif dataset_name == "go_emotions":
            # Ищем в GoEmotions
            emotion_id = None
            for i, e in enumerate(dataset["train"].features["labels"].feature.names):
                if e.lower() == emotion.lower():
                    emotion_id = i
                    break
                    
            if emotion_id is not None:
                for item in dataset["train"]:
                    if emotion_id in item["labels"]:
                        examples.append(item["text"])
                        if len(examples) >= limit:
                            break
                            
        return examples

# Класс для загрузки и интеграции KnowledgeBases
class KnowledgeBaseIntegrator:
    def __init__(self):
        self.loaded_kbs = {}
        
    async def load_conceptnet(self, language: str = "en"):
        """
        Загрузить и подготовить ConceptNet
        """
        try:
            import networkx as nx
            import requests
            
            if "conceptnet" not in self.loaded_kbs:
                self.loaded_kbs["conceptnet"] = {
                    "graph": nx.DiGraph(),
                    "cache": {}
                }
                
            logger.info("ConceptNet API ready for queries")
            return self.loaded_kbs["conceptnet"]
        except Exception as e:
            logger.error(f"Error setting up ConceptNet: {e}")
            return None
            
    async def query_conceptnet(self, concept: str, relation: str = None, limit: int = 10):
        """
        Запрос к ConceptNet API
        """
        if "conceptnet" not in self.loaded_kbs:
            await self.load_conceptnet()
            
        cache_key = f"{concept}_{relation}_{limit}"
        if cache_key in self.loaded_kbs["conceptnet"]["cache"]:
            return self.loaded_kbs["conceptnet"]["cache"][cache_key]
            
        try:
            import requests
            
            # Формируем URL запроса
            concept = concept.lower().replace(" ", "_")
            url = f"http://api.conceptnet.io/c/en/{concept}"
            if relation:
                url = f"http://api.conceptnet.io/query?node=/c/en/{concept}&rel=/r/{relation}&limit={limit}"
                
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                self.loaded_kbs["conceptnet"]["cache"][cache_key] = data
                return data
            else:
                logger.error(f"ConceptNet API error: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error querying ConceptNet: {e}")
            return None
    
    async def load_wordnet(self):
        """
        Загрузить и подготовить WordNet
        """
        try:
            from nltk.corpus import wordnet
            import nltk
            
            # Проверяем, загружен ли уже WordNet
            try:
                wordnet.synsets("test")
            except LookupError:
                nltk.download('wordnet')
                
            self.loaded_kbs["wordnet"] = True
            logger.info("WordNet loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading WordNet: {e}")
            return False
            
    async def query_wordnet(self, word: str, pos: str = None):
        """
        Запрос к WordNet
        """
        if "wordnet" not in self.loaded_kbs:
            success = await self.load_wordnet()
            if not success:
                return None
                
        try:
            from nltk.corpus import wordnet as wn
            
            # Маппинг частей речи
            pos_map = {"n": wn.NOUN, "v": wn.VERB, "a": wn.ADJ, "r": wn.ADV}
            
            if pos and pos in pos_map:
                synsets = wn.synsets(word, pos=pos_map[pos])
            else:
                synsets = wn.synsets(word)
                
            result = []
            for synset in synsets:
                synonyms = [lemma.name() for lemma in synset.lemmas()]
                hypernyms = [h.name().split('.')[0] for h in synset.hypernyms()]
                
                result.append({
                    "synset": synset.name(),
                    "definition": synset.definition(),
                    "synonyms": synonyms,
                    "hypernyms": hypernyms,
                    "pos": synset.pos()
                })
                
            return result
        except Exception as e:
            logger.error(f"Error querying WordNet: {e}")
            return None

# Multimodal emotion analysis with event bus integration
class MultimodalEmotionAnalyzer:
    def __init__(self, nats_url: str = "nats://localhost:4222"):
        self.text_analyzer = TextEmotionAnalyzer()
        self.audio_analyzer = AudioEmotionAnalyzer()
        self.video_analyzer = VideoEmotionAnalyzer()
        self.publisher = EmotionEventPublisher(nats_url)
        self.datasets = EmotionalDatasetLoader()
        self.knowledge = KnowledgeBaseIntegrator()
        
    async def analyze(self, 
                     text: Optional[str] = None, 
                     audio_data: Optional[Union[str, np.ndarray]] = None, 
                     video_data: Optional[Union[str, np.ndarray]] = None,
                     publish_events: bool = True) -> Dict[str, Any]:
        """
        Комплексный анализ эмоций по разным модальностям
        """
        results = {}
        tasks = []
        
        if text:
            tasks.append(self.text_analyzer.analyze(text))
        if audio_data is not None:
            tasks.append(self.audio_analyzer.analyze(audio_data))
        if video_data is not None:
            tasks.append(self.video_analyzer.analyze(video_data))
            
        if not tasks:
            return {"error": "No input data provided"}
            
        # Запускаем анализ по всем модальностям параллельно
        modal_results = await asyncio.gather(*tasks)
        
        # Заполняем результаты по модальностям
        if text:
            results["text"] = modal_results.pop(0)
        if audio_data is not None:
            results["audio"] = modal_results.pop(0)
        if video_data is not None:
            results["video"] = modal_results.pop(0)
            
        # Определяем основную эмоцию (с приоритетом видео > аудио > текст)
        main_emotion = None
        main_score = 0.0
        
        for modality in ["video", "audio", "text"]:
            if modality in results and "error" not in results[modality]:
                emotion = results[modality]["emotion"]
                score = results[modality]["score"]
                
                # Если еще нет основной эмоции или текущая более выражена
                if main_emotion is None or score > main_score:
                    main_emotion = emotion
                    main_score = score
                    
        results["dominant_emotion"] = main_emotion or "neutral"
        results["dominant_score"] = main_score
        results["timestamp"] = asyncio.get_event_loop().time()
        
        # Публикуем событие в event bus
        if publish_events:
            await self.publisher.publish_emotion("sasok.emotion.analysis", results)
            
        return results
    
    async def get_emotion_examples(self, emotion: str) -> Dict[str, Any]:
        """
        Получить примеры из датасетов для данной эмоции
        """
        empathetic = await self.datasets.get_examples_by_emotion(
            emotion, "empathetic_dialogues", 3)
        go_emotions = await self.datasets.get_examples_by_emotion(
            emotion, "go_emotions", 3)
            
        return {
            "empathetic_dialogues": empathetic,
            "go_emotions": go_emotions
        }
    
    async def get_emotion_knowledge(self, emotion: str) -> Dict[str, Any]:
        """
        Получить знания об эмоции из knowledge bases
        """
        conceptnet = await self.knowledge.query_conceptnet(emotion)
        wordnet = await self.knowledge.query_wordnet(emotion)
        
        return {
            "conceptnet": conceptnet,
            "wordnet": wordnet
        }
        
    async def close(self):
        """
        Закрыть все соединения
        """
        await self.publisher.close()
