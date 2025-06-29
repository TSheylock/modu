import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# API Configuration
API_CONFIG = {
    "title": "SASOK Platform API",
    "description": "Backend API for SASOK Intelligence Platform",
    "version": "1.0.0",
    "docs_url": "/docs",
    "redoc_url": "/redoc"
}

# Security Configuration
SECURITY_CONFIG = {
    "secret_key": os.getenv("SECRET_KEY", "your-secret-key-here"),
    "algorithm": "HS256",
    "access_token_expire_minutes": 30
}

# Web3 Configuration
WEB3_CONFIG = {
    "infura_project_id": os.getenv("INFURA_PROJECT_ID", ""),
    "network": os.getenv("WEB3_NETWORK", "mainnet"),
    "contract_addresses": {
        "main": os.getenv("MAIN_CONTRACT_ADDRESS", ""),
        "nft": os.getenv("NFT_CONTRACT_ADDRESS", "")
    }
}

# AI Model Configuration
AI_CONFIG = {
    "emotion_model": "j-hartmann/emotion-english-distilroberta-base",
    "nlp_model": "facebook/bart-large-mnli",
    "image_model": "microsoft/resnet-50",
    "batch_size": 32,
    "max_sequence_length": 512
}

# Analytics Configuration
ANALYTICS_CONFIG = {
    "storage_type": "memory",  # Options: memory, database
    "retention_days": 30,
    "batch_size": 100
}

# Database Configuration
DATABASE_CONFIG = {
    "url": os.getenv("DATABASE_URL", "sqlite:///./sasok.db"),
    "min_connections": 1,
    "max_connections": 10
}

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO"
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "sasok.log",
            "formatter": "default",
            "level": "DEBUG"
        }
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO"
    }
}

# CORS Configuration
CORS_CONFIG = {
    "allow_origins": [
        "http://localhost:8000",
        "http://localhost:3000"
    ],
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"]
}

# Cache Configuration
CACHE_CONFIG = {
    "type": "memory",
    "ttl": 300,  # 5 minutes
    "max_size": 1000
}

class Config:
    """Configuration class to manage all settings"""
    
    def __init__(self):
        self.base_dir = BASE_DIR
        self.api = API_CONFIG
        self.security = SECURITY_CONFIG
        self.web3 = WEB3_CONFIG
        self.ai = AI_CONFIG
        self.analytics = ANALYTICS_CONFIG
        self.database = DATABASE_CONFIG
        self.logging = LOGGING_CONFIG
        self.cors = CORS_CONFIG
        self.cache = CACHE_CONFIG

    def get_settings(self) -> Dict[str, Any]:
        """Get all settings as a dictionary"""
        return {
            "base_dir": self.base_dir,
            "api": self.api,
            "security": self.security,
            "web3": self.web3,
            "ai": self.ai,
            "analytics": self.analytics,
            "database": self.database,
            "logging": self.logging,
            "cors": self.cors,
            "cache": self.cache
        }

    def update_from_env(self):
        """Update configuration from environment variables"""
        # Update Web3 settings
        if os.getenv("INFURA_PROJECT_ID"):
            self.web3["infura_project_id"] = os.getenv("INFURA_PROJECT_ID")
        
        # Update security settings
        if os.getenv("SECRET_KEY"):
            self.security["secret_key"] = os.getenv("SECRET_KEY")
        
        # Update database settings
        if os.getenv("DATABASE_URL"):
            self.database["url"] = os.getenv("DATABASE_URL")

# Create global config instance
config = Config()
