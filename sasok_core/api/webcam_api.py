"""
FastAPI endpoints for WebRTC video stream processing and real-time emotion analysis
Supports multimodal analysis (video, audio, text) with NATS event integration
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends, Request, Body
from fastapi.security import APIKeyHeader
from typing import Dict, Any, Optional, List, Union
import asyncio
import base64
import numpy as np
import cv2
import json
import logging
import time
from io import BytesIO
from pathlib import Path
import os

# Импорт обновленного модуля анализа эмоций
from sasok_core.modules.emotion.emotion_analysis import (
    MultimodalEmotionAnalyzer, 
    VideoEmotionAnalyzer,
    TextEmotionAnalyzer,
    AudioEmotionAnalyzer,
    KnowledgeBaseIntegrator
)

router = APIRouter(prefix="/emotion")

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WebcamAPI")

# Инициализация анализаторов
emotion_analyzer = MultimodalEmotionAnalyzer()
video_analyzer = VideoEmotionAnalyzer()
text_analyzer = TextEmotionAnalyzer()
audio_analyzer = AudioEmotionAnalyzer()
knowledge_base = KnowledgeBaseIntegrator()

# Временная директория для хранения кадров
TEMP_DIR = Path("/tmp/sasok_webcam_frames")
TEMP_DIR.mkdir(exist_ok=True)

# Защита API ключом
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

API_KEYS = {
    "sasok_camera": "sasok-emotion-key-2025"
}

async def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
    if api_key not in API_KEYS.values():
        raise HTTPException(
            status_code=403, 
            detail="Invalid API key"
        )
    return api_key

# Расширенная функция для анализа кадра (добавлены метаданные)
async def analyze_frame(frame_data: np.ndarray, session_id: str = None, metadata: Dict = None) -> Dict[str, Any]:
    try:
        # Анализ эмоций на кадре
        result = await video_analyzer.analyze(frame_data)
        
        # Добавляем метаданные
        if result and not "error" in result:
            result["session_id"] = session_id
            result["metadata"] = metadata or {}
            result["timestamp"] = time.time()
            
            # Публикуем в event bus
            await emotion_analyzer.publisher.publish_emotion(
                "sasok.emotion.realtime.video", result
            )
            
        return result
    except Exception as e:
        logger.error(f"Error analyzing frame: {e}")
        return {
            "error": str(e),
            "timestamp": time.time()
        }

@router.websocket("/ws/video-emotion")
async def video_emotion_ws(websocket: WebSocket):
    """
    WebSocket endpoint для обработки видеопотока с камеры в реальном времени
    """
    await websocket.accept()
    session_id = f"session_{int(time.time())}"
    frame_count = 0
    
    try:
        # Инициализация сессии
        logger.info(f"Starting new video emotion session: {session_id}")
        
        # Основной цикл обработки
        while True:
            # Получаем данные от клиента
            data = await websocket.receive_json()
            frame_base64 = data.get("frame")
            metadata = data.get("metadata", {})
            
            if not frame_base64:
                await websocket.send_json({"error": "No frame data received"})
                continue
                
            # Декодируем base64 в numpy array
            try:
                frame_bytes = base64.b64decode(frame_base64.split(',')[1] if ',' in frame_base64 else frame_base64)
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None or frame.size == 0:
                    await websocket.send_json({"error": "Invalid frame data"})
                    continue
                    
                # Анализ эмоций
                frame_count += 1
                result = await analyze_frame(frame, session_id, {
                    **metadata,
                    "frame_count": frame_count
                })
                
                # Отправка результата обратно клиенту
                await websocket.send_json(result)
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                await websocket.send_json({"error": f"Processing error: {str(e)}"})
                
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from video session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Завершение сессии
        logger.info(f"Ending video emotion session: {session_id}, processed {frame_count} frames")

@router.post("/analyze-text", dependencies=[Depends(verify_api_key)])
async def analyze_text(request: Dict[str, Any] = Body(...)):
    """
    Эндпоинт для анализа текста на эмоции
    """
    text = request.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
        
    result = await text_analyzer.analyze(text)
    
    # Публикуем в event bus
    await emotion_analyzer.publisher.publish_emotion(
        "sasok.emotion.realtime.text", result
    )
    
    return result

@router.post("/analyze-audio", dependencies=[Depends(verify_api_key)])
async def analyze_audio(request: Dict[str, Any] = Body(...)):
    """
    Эндпоинт для анализа аудио на эмоции
    """
    audio_base64 = request.get("audio")
    if not audio_base64:
        raise HTTPException(status_code=400, detail="No audio provided")
        
    # Декодируем base64 в numpy array для аудио
    audio_bytes = base64.b64decode(audio_base64.split(',')[1] if ',' in audio_base64 else audio_base64)
    
    # Временный файл для аудио
    temp_audio_path = TEMP_DIR / f"audio_{int(time.time())}.wav"
    
    try:
        with open(temp_audio_path, "wb") as f:
            f.write(audio_bytes)
            
        result = await audio_analyzer.analyze(str(temp_audio_path))
        
        # Публикуем в event bus
        await emotion_analyzer.publisher.publish_emotion(
            "sasok.emotion.realtime.audio", result
        )
        
        # Удаляем временный файл
        os.remove(temp_audio_path)
        
        return result
    except Exception as e:
        logger.error(f"Error analyzing audio: {e}")
        if temp_audio_path.exists():
            os.remove(temp_audio_path)
        raise HTTPException(status_code=500, detail=f"Audio analysis error: {str(e)}")

@router.post("/multimodal", dependencies=[Depends(verify_api_key)])
async def analyze_multimodal(request: Dict[str, Any] = Body(...)):
    """
    Эндпоинт для комплексного мультимодального анализа (текст + аудио + видео)
    """
    text = request.get("text")
    audio_base64 = request.get("audio")
    video_base64 = request.get("video")
    
    if not any([text, audio_base64, video_base64]):
        raise HTTPException(status_code=400, detail="No data provided for analysis")
    
    # Подготовка данных для анализа
    audio_data = None
    video_data = None
    
    try:
        # Обработка аудио если есть
        if audio_base64:
            audio_bytes = base64.b64decode(audio_base64.split(',')[1] if ',' in audio_base64 else audio_base64)
            temp_audio_path = TEMP_DIR / f"audio_mm_{int(time.time())}.wav"
            with open(temp_audio_path, "wb") as f:
                f.write(audio_bytes)
            audio_data = str(temp_audio_path)
        
        # Обработка видео если есть
        if video_base64:
            video_bytes = base64.b64decode(video_base64.split(',')[1] if ',' in video_base64 else video_base64)
            nparr = np.frombuffer(video_bytes, np.uint8)
            video_data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Мультимодальный анализ
        result = await emotion_analyzer.analyze(
            text=text,
            audio_data=audio_data,
            video_data=video_data,
            publish_events=True
        )
        
        # Очистка временных файлов
        if audio_data and isinstance(audio_data, str) and Path(audio_data).exists():
            os.remove(audio_data)
            
        return result
        
    except Exception as e:
        logger.error(f"Error in multimodal analysis: {e}")
        # Очистка в случае ошибки
        if audio_data and isinstance(audio_data, str) and Path(audio_data).exists():
            os.remove(audio_data)
        raise HTTPException(status_code=500, detail=f"Multimodal analysis error: {str(e)}")

@router.get("/knowledge/{emotion}", dependencies=[Depends(verify_api_key)])
async def get_emotion_knowledge(emotion: str):
    """
    Эндпоинт для получения знаний об эмоции из баз знаний
    """
    if not emotion:
        raise HTTPException(status_code=400, detail="No emotion specified")
        
    try:
        # Получаем знания из ConceptNet и WordNet
        kb_data = await emotion_analyzer.get_emotion_knowledge(emotion)
        
        # Получаем примеры из датасетов
        examples = await emotion_analyzer.get_emotion_examples(emotion)
        
        return {
            "knowledge": kb_data,
            "examples": examples,
            "emotion": emotion
        }
    except Exception as e:
        logger.error(f"Error getting emotion knowledge: {e}")
        raise HTTPException(status_code=500, detail=f"Knowledge base error: {str(e)}")
