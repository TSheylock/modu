"""
Модуль анализа эмоций по видео с веб-камеры

Архетипический анализ эмоций через интеграцию с MediaPipe и DeepFace
для создания метакогнитивного слоя Sense Layer в SASOK.
"""
import os
import cv2
import time
import json
import asyncio
import logging
import threading
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Путь импорта будет заменен на реальный после установки DeepFace
try:
    from deepface import DeepFace
except ImportError:
    print("DeepFace не установлен. Будет использован заглушка.")
    # Заглушка для DeepFace при отсутствии
    class DeepFaceMock:
        @staticmethod
        def analyze(img_path, actions=None, enforce_detection=True, silent=False):
            return [{"emotion": {"angry": 0, "disgust": 0, "fear": 0, 
                                "happy": 0, "sad": 0, "surprise": 0, 
                                "neutral": 100}}]
    DeepFace = DeepFaceMock()

from sasok_core.perception.emotion_base import EmotionAnalyzer
from sasok_core.utils.nats_client import NatsClient

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SASOK.Perception.WebcamEmotion")

# Константы модуля
MODELS_DIR = os.getenv("MODELS_DIR", "models")
FRAME_BUFFER_SIZE = 30  # Количество кадров для буферизации
ANALYSIS_INTERVAL = 1.0  # Интервал анализа в секундах
CONFIDENCE_THRESHOLD = 0.7  # Порог уверенности для меток SASOK_DOUBT
DATA_ENCRYPTION_KEY = os.getenv("DATA_ENCRYPTION_KEY", "sasok_secure_key")

class WebcamEmotionAnalyzer(EmotionAnalyzer):
    """
    Анализатор эмоций по видео с веб-камеры
    Реализует принцип мультиуровневого анализа SASOK
    """
    
    def __init__(self):
        """
        Инициализация анализатора эмоций
        """
        super().__init__("webcam")
        
        # Инициализация MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Инициализация параметров
        self.camera_id = 0
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 30
        
        # Буфер кадров для анализа
        self.frame_buffer = []
        self.lock = threading.Lock()
        
        # Результаты анализа
        self.last_emotion = None
        self.emotion_history = []
        self.last_analysis_time = 0
        
        # Флаг работы
        self.running = False
        
        # Инициализация NATS клиента
        self.nats_client = NatsClient()
        
        # Загрузка моделей
        self._load_models()
        
        logger.info("WebcamEmotionAnalyzer инициализирован")
    
    def _load_models(self):
        """
        Загрузка моделей для анализа эмоций
        """
        try:
            # Подготовка MediaPipe Face Mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Подготовка DeepFace для анализа эмоций
            self.emotion_models = [
                'emotion', 
                'age', 
                'gender', 
                'race'
            ]
            
            # Предзагрузка моделей с минимальным изображением для ускорения первого анализа
            try:
                _ = DeepFace.analyze(
                    img_path=np.zeros((100, 100, 3), dtype=np.uint8),
                    actions=self.emotion_models,
                    enforce_detection=False,
                    silent=True
                )
                logger.info("Модели DeepFace предзагружены")
            except Exception as e:
                logger.warning(f"Предзагрузка DeepFace не удалась: {e}")
                if "SASOK_DOUBT" not in self.tags:
                    self.tags.append("SASOK_DOUBT")
            
            logger.info("Модели для анализа эмоций загружены")
        except Exception as e:
            logger.error(f"Ошибка загрузки моделей: {e}")
            if "SASOK_DOUBT" not in self.tags:
                self.tags.append("SASOK_DOUBT")
    
    async def initialize(self):
        """
        Асинхронная инициализация аналитических компонентов
        """
        try:
            # Подключение к NATS
            await self.nats_client.connect()
            
            # Подписка на события
            await self.nats_client.subscribe(
                "sasok.webcam.command",
                self._handle_webcam_command
            )
            
            logger.info("Компоненты анализа инициализированы")
            return True
        except Exception as e:
            logger.error(f"Ошибка инициализации: {e}")
            return False
    
    async def analyze(self, data):
        """
        Анализ данных (переопределение метода базового класса)
        
        Args:
            data: Кадр изображения для анализа
            
        Returns:
            Dict: Результат анализа эмоций
        """
        if isinstance(data, np.ndarray):
            # Это кадр из видеопотока
            result = self._analyze_frame(data)
            if result:
                return await self.format_result(result["emotions"], result)
        
        return await self.format_result({"SASOK_DOUBT": True}, {"error": "Неподдерживаемый тип данных"})
    
    async def _handle_webcam_command(self, msg):
        """
        Обработчик команд для веб-камеры
        """
        try:
            data = json.loads(msg.data.decode())
            command = data.get("command")
            
            if command == "start":
                await self.start_webcam()
            elif command == "stop":
                await self.stop_webcam()
            elif command == "configure":
                config = data.get("config", {})
                await self.configure(config)
            else:
                logger.warning(f"Неизвестная команда: {command}")
        except Exception as e:
            logger.error(f"Ошибка обработки команды: {e}")
    
    async def start_webcam(self):
        """
        Запуск анализа с веб-камеры
        """
        if self.running:
            logger.warning("Анализ с веб-камеры уже запущен")
            return
        
        try:
            self.running = True
            
            # Запуск захвата видео в отдельном потоке
            threading.Thread(
                target=self._capture_thread, 
                daemon=True
            ).start()
            
            # Запуск анализа в отдельном потоке
            threading.Thread(
                target=self._analysis_thread, 
                daemon=True
            ).start()
            
            # Публикация события о запуске
            await self.nats_client.publish(
                "sasok.webcam.status",
                json.dumps({"status": "started"})
            )
            
            logger.info("Анализ с веб-камеры запущен")
            return True
        except Exception as e:
            self.running = False
            logger.error(f"Ошибка запуска анализа: {e}")
            return False
    
    async def stop_webcam(self):
        """
        Остановка анализа с веб-камеры
        """
        if not self.running:
            logger.warning("Анализ с веб-камеры не запущен")
            return
        
        try:
            self.running = False
            
            # Публикация события об остановке
            await self.nats_client.publish(
                "sasok.webcam.status",
                json.dumps({"status": "stopped"})
            )
            
            logger.info("Анализ с веб-камеры остановлен")
            return True
        except Exception as e:
            logger.error(f"Ошибка остановки анализа: {e}")
            return False
    
    async def configure(self, config):
        """
        Конфигурация параметров анализа
        """
        try:
            if "camera_id" in config:
                self.camera_id = config["camera_id"]
            
            if "frame_width" in config:
                self.frame_width = config["frame_width"]
            
            if "frame_height" in config:
                self.frame_height = config["frame_height"]
            
            if "fps" in config:
                self.fps = config["fps"]
            
            # Публикация события о конфигурации
            await self.nats_client.publish(
                "sasok.webcam.status",
                json.dumps({"status": "configured", "config": config})
            )
            
            logger.info(f"Параметры анализа сконфигурированы: {config}")
            return True
        except Exception as e:
            logger.error(f"Ошибка конфигурации: {e}")
            return False
    
    def _capture_thread(self):
        """
        Поток захвата видео с веб-камеры
        """
        try:
            cap = cv2.VideoCapture(self.camera_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            if not cap.isOpened():
                logger.error("Не удалось открыть веб-камеру")
                self.running = False
                return
            
            logger.info(f"Захват видео запущен: камера {self.camera_id}")
            
            while self.running:
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning("Не удалось захватить кадр")
                    continue
                
                # Добавление кадра в буфер (с локальной обработкой - приватность)
                with self.lock:
                    self.frame_buffer.append({
                        "frame": frame.copy(),
                        "timestamp": time.time()
                    })
                    
                    # Ограничение размера буфера
                    while len(self.frame_buffer) > FRAME_BUFFER_SIZE:
                        self.frame_buffer.pop(0)
                
                # Задержка для соблюдения FPS
                time.sleep(1.0 / self.fps)
            
            cap.release()
            logger.info("Захват видео остановлен")
        except Exception as e:
            logger.error(f"Ошибка в потоке захвата: {e}")
            self.running = False
    
    def _analysis_thread(self):
        """
        Поток анализа эмоций
        """
        try:
            logger.info("Поток анализа эмоций запущен")
            
            while self.running:
                # Проверка интервала анализа
                current_time = time.time()
                if current_time - self.last_analysis_time < ANALYSIS_INTERVAL:
                    time.sleep(0.1)
                    continue
                
                self.last_analysis_time = current_time
                
                # Получение последнего кадра из буфера
                with self.lock:
                    if not self.frame_buffer:
                        time.sleep(0.1)
                        continue
                    
                    frame_data = self.frame_buffer[-1]
                    frame = frame_data["frame"]
                    timestamp = frame_data["timestamp"]
                
                # Анализ эмоций
                emotions = self._analyze_frame(frame)
                
                if emotions:
                    # Сохранение результатов
                    self.last_emotion = emotions
                    self.emotion_history.append({
                        "timestamp": timestamp,
                        "emotions": emotions
                    })
                    
                    # Ограничение истории
                    while len(self.emotion_history) > 100:
                        self.emotion_history.pop(0)
                    
                    # Публикация события об эмоциях
                    asyncio.run(self._publish_emotion_event(emotions, frame))
                
                # Задержка для соблюдения интервала анализа
                time.sleep(max(0, ANALYSIS_INTERVAL - (time.time() - current_time)))
            
            logger.info("Поток анализа эмоций остановлен")
        except Exception as e:
            logger.error(f"Ошибка в потоке анализа: {e}")
            self.running = False
    
    def _analyze_frame(self, frame):
        """
        Анализ эмоций на кадре
        
        Args:
            frame: Кадр для анализа
            
        Returns:
            Словарь с эмоциями и их значениями
        """
        try:
            # Конвертация BGR в RGB для MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Анализ лица с помощью MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            # Если лицо не найдено
            if not results.multi_face_landmarks:
                logger.debug("Лицо не обнаружено")
                return None
            
            # Анализ эмоций с помощью DeepFace
            face_analysis = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            if not face_analysis:
                logger.debug("DeepFace не смог проанализировать эмоции")
                return None
            
            # Получение эмоций
            emotions = face_analysis[0].get("emotion", {})
            
            # Преобразование к русским названиям и нормализация
            emotion_map = {
                "angry": "злость",
                "disgust": "отвращение",
                "fear": "страх",
                "happy": "радость",
                "sad": "грусть",
                "surprise": "удивление",
                "neutral": "нейтральность"
            }
            
            normalized_emotions = {}
            for eng_name, value in emotions.items():
                ru_name = emotion_map.get(eng_name, eng_name)
                normalized_emotions[ru_name] = value / 100.0  # Нормализация 0-1
            
            # Проверка уверенности
            max_emotion = max(normalized_emotions.items(), key=lambda x: x[1])
            if max_emotion[1] < CONFIDENCE_THRESHOLD:
                normalized_emotions["SASOK_DOUBT"] = True
            
            # Добавление метаданных Face Mesh
            # Использование первого обнаруженного лица
            face_landmarks = results.multi_face_landmarks[0]
            
            # Извлечение ключевых точек для анализа
            landmarks_dict = {}
            for idx, landmark in enumerate(face_landmarks.landmark):
                landmarks_dict[f"landmark_{idx}"] = {
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z
                }
            
            # Расчет метрик лица
            metrics = self._calculate_face_metrics(face_landmarks)
            
            # Формирование итогового результата
            result = {
                "emotions": normalized_emotions,
                "metrics": metrics,
                "landmarks": landmarks_dict,
                "timestamp": time.time()
            }
            
            return result
        except Exception as e:
            logger.error(f"Ошибка анализа кадра: {e}")
            return None
    
    def _calculate_face_metrics(self, face_landmarks):
        """
        Расчет метрик лица на основе ключевых точек
        
        Args:
            face_landmarks: Ключевые точки лица
            
        Returns:
            Словарь с метриками
        """
        try:
            # Индексы для ключевых точек (MediaPipe Face Mesh)
            # Глаза
            left_eye = [33, 133]
            right_eye = [362, 263]
            
            # Рот
            upper_lip = 13
            lower_lip = 14
            left_corner = 61
            right_corner = 291
            
            # Брови
            left_eyebrow = [70, 63]
            right_eyebrow = [300, 293]
            
            # Вычисление метрик
            landmarks = face_landmarks.landmark
            
            # Открытость глаз (расстояние между верхним и нижним веком)
            left_eye_open = _calculate_distance(
                landmarks[left_eye[0]], landmarks[left_eye[1]]
            )
            right_eye_open = _calculate_distance(
                landmarks[right_eye[0]], landmarks[right_eye[1]]
            )
            
            # Открытость рта (расстояние между верхней и нижней губой)
            mouth_open = _calculate_distance(
                landmarks[upper_lip], landmarks[lower_lip]
            )
            
            # Ширина рта (расстояние между уголками)
            mouth_width = _calculate_distance(
                landmarks[left_corner], landmarks[right_corner]
            )
            
            # Поднятие бровей (расстояние между бровями и глазами)
            left_brow_height = _calculate_distance(
                landmarks[left_eyebrow[0]], landmarks[left_eye[0]]
            )
            right_brow_height = _calculate_distance(
                landmarks[right_eyebrow[0]], landmarks[right_eye[0]]
            )
            
            # Формирование метрик
            metrics = {
                "left_eye_open": left_eye_open,
                "right_eye_open": right_eye_open,
                "eye_open_ratio": (left_eye_open + right_eye_open) / 2,
                "mouth_open": mouth_open,
                "mouth_width": mouth_width,
                "mouth_aspect_ratio": mouth_open / mouth_width if mouth_width > 0 else 0,
                "left_brow_height": left_brow_height,
                "right_brow_height": right_brow_height,
                "brow_height_ratio": (left_brow_height + right_brow_height) / 2
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Ошибка вычисления метрик лица: {e}")
            return {}
    
    async def _publish_emotion_event(self, emotions, frame=None):
        """
        Публикация события об эмоциях
        
        Args:
            emotions: Словарь с эмоциями
            frame: Кадр с отметками (опционально)
        """
        try:
            # Формирование данных события
            event_data = {
                "source": "webcam",
                "emotions": emotions["emotions"],
                "metrics": emotions["metrics"],
                "timestamp": emotions["timestamp"],
                "has_image": frame is not None
            }
            
            # Публикация события
            await self.nats_client.publish(
                "sasok.emotion.detected",
                json.dumps(event_data)
            )
            
            # Если есть кадр, сохраняем его и публикуем отдельно
            if frame is not None:
                # Рисование меток на кадре
                annotated_frame = self._draw_annotations(frame, emotions)
                
                # Кодирование кадра в JPEG
                success, encoded_img = cv2.imencode('.jpg', annotated_frame)
                
                if success:
                    # Публикация изображения
                    await self.nats_client.publish(
                        "sasok.emotion.image",
                        encoded_img.tobytes()
                    )
            
            logger.debug(f"Событие об эмоциях опубликовано: {event_data['emotions']}")
        except Exception as e:
            logger.error(f"Ошибка публикации события: {e}")
    
    def _draw_annotations(self, frame, emotions):
        """
        Рисование аннотаций на кадре
        
        Args:
            frame: Исходный кадр
            emotions: Данные о эмоциях
            
        Returns:
            Кадр с аннотациями
        """
        try:
            # Создание копии кадра
            result = frame.copy()
            
            # Информация об эмоциях
            emotion_text = ""
            for emotion, value in emotions["emotions"].items():
                if emotion == "SASOK_DOUBT":
                    continue
                emotion_text += f"{emotion}: {value:.2f}, "
            
            # Цвет текста в зависимости от уверенности
            text_color = (0, 255, 0)  # Зеленый - высокая уверенность
            if "SASOK_DOUBT" in emotions["emotions"]:
                text_color = (0, 165, 255)  # Оранжевый - низкая уверенность
            
            # Добавление текста с эмоциями
            cv2.putText(
                result,
                emotion_text.rstrip(", "),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                2
            )
            
            return result
        except Exception as e:
            logger.error(f"Ошибка рисования аннотаций: {e}")
            return frame

def _calculate_distance(point1, point2):
    """
    Вычисление евклидова расстояния между двумя точками
    
    Args:
        point1: Первая точка
        point2: Вторая точка
        
    Returns:
        Расстояние между точками
    """
    return ((point1.x - point2.x) ** 2 + 
            (point1.y - point2.y) ** 2 + 
            (point1.z - point2.z) ** 2) ** 0.5

# Тестовый код
if __name__ == "__main__":
    import asyncio
    
    async def main():
        analyzer = WebcamEmotionAnalyzer()
        await analyzer.initialize()
        await analyzer.start_webcam()
        
        try:
            print("Анализ запущен. Нажмите Ctrl+C для завершения.")
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("Завершение работы...")
        finally:
            await analyzer.stop_webcam()
    
    asyncio.run(main())
