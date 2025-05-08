"""
Главный API модуль SASOK

Объединяет все API эндпоинты и обеспечивает настройки безопасности,
CORS и глобальные обработчики ошибок
"""
import os
import logging
import json
import time
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, Request, Response, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN, HTTP_500_INTERNAL_SERVER_ERROR

# Загрузка переменных окружения
load_dotenv()

# Настройка логгера
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SASOK.API")

# Импорт роутеров
from sasok_core.api.auth import router as auth_router
from sasok_core.api.emotion_api import router as emotion_router
from sasok_core.api.knowledge_api import router as knowledge_router
from sasok_core.api.graph_api import router as graph_router

# Создание FastAPI приложения
app = FastAPI(
    title="SASOK API",
    description="API для платформы анализа эмоций SASOK",
    version="0.1.0"
)

# Настройка CORS
origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API ключи для защиты эндпоинтов
API_KEYS = {
    "emotion": os.getenv("API_KEY_EMOTION", "sasok-emotion-key-2025"),
    "analysis": os.getenv("API_KEY_ANALYSIS", "sasok-analysis-key-2025")
}

# Настройка проверки API ключа
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
    """
    Проверка API ключа
    """
    if api_key not in API_KEYS.values():
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    return api_key

# Проверка работоспособности NATS
async def check_nats_availability():
    """
    Проверка доступности NATS сервера
    """
    from sasok_core.utils.nats_client import NatsClient
    
    nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
    
    try:
        client = await NatsClient.get_instance(nats_url)
        if client.connected:
            logger.info(f"NATS сервер доступен: {nats_url}")
            return True
        else:
            logger.warning(f"NATS сервер недоступен: {nats_url}")
            return False
    except Exception as e:
        logger.error(f"Ошибка проверки NATS: {e}")
        return False

# Включение роутеров
app.include_router(auth_router)
app.include_router(emotion_router)
app.include_router(knowledge_router)
app.include_router(graph_router)

# Подключение статических файлов
static_files_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_files_dir)), name="static")

# Корневой эндпоинт
@app.get("/", response_class=HTMLResponse)
async def root():
    return RedirectResponse(url="/static/index.html")

@app.get("/api")
async def api_root():
    return {
        "message": "SASOK Core API",
        "version": "0.1.0",
        "status": "active"
    }

# Эндпоинт для проверки состояния системы
@app.get("/health", tags=["System"])
async def health_check():
    # Проверка состояния компонентов
    nats_status = await check_nats_availability()
    
    # Проверка наличия моделей
    models_dir = Path("/home/sasok/Рабочий стол/blackboxai-1745739396945/models")
    models_ready = models_dir.exists()
    
    return {
        "status": "healthy" if nats_status and models_ready else "degraded",
        "components": {
            "api": "up",
            "nats": "up" if nats_status else "down",
            "models": "ready" if models_ready else "not_ready"
        },
        "timestamp": time.time()
    }

# Middleware для логирования запросов
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Логируем запрос
    logger.info(f"Request: {request.method} {request.url}")
    
    try:
        response = await call_next(request)
        
        # Логируем ответ
        process_time = time.time() - start_time
        logger.info(f"Response: {response.status_code} (took {process_time:.3f}s)")
        
        return response
    except Exception as e:
        # Логируем ошибки
        process_time = time.time() - start_time
        logger.error(f"Error: {str(e)} (after {process_time:.3f}s)")
        
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal Server Error"}
        )

# Обработчик событий запуска
@app.on_event("startup")
async def startup_event():
    logger.info("Запуск API SASOK")
    
    # Проверка NATS
    nats_status = await check_nats_availability()
    if not nats_status:
        logger.warning("NATS недоступен. Некоторые функции могут работать некорректно.")
    
    # Проверка переменных окружения
    required_env_vars = [
        "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD",
        "INFURA_API_KEY", "ETHEREUM_NETWORK"
    ]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Отсутствуют переменные окружения: {', '.join(missing_vars)}")

# Обработчик событий остановки
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Остановка API SASOK")
    
    # Закрытие соединений
    try:
        from sasok_core.utils.nats_client import NatsClient
        client = await NatsClient.get_instance()
        await client.close()
    except Exception as e:
        logger.error(f"Ошибка при закрытии NATS: {e}")

# Запуск сервера при прямом вызове
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "sasok_core.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
