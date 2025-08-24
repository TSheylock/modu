"""
Web3 клиент для SASOK

Обеспечивает безопасное взаимодействие с блокчейном Ethereum,
интеграцию с EmotionalSBT и управление Zero-Knowledge доказательствами
"""
import os
import json
import time
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
from dotenv import load_dotenv
from web3 import Web3
from web3.middleware import geth_poa_middleware
import hashlib
import random

# Настройка логгера
logger = logging.getLogger("Web3Client")

# Загрузка переменных окружения
load_dotenv()

class Web3Client:
    """
    Клиент для взаимодействия с блокчейном Ethereum и смарт-контрактами
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """
        Синглтон для получения единого экземпляра клиента
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """
        Инициализация Web3 клиента
        """
        # Получение API ключа Infura из переменных окружения
        self.infura_api_key = os.getenv("INFURA_API_KEY", "gCtOf0p4fM2811/e0i5kV+jDdMyd3xZ2SOCPnm+B0B0vGYgBNk0wjA")
        self.infura_project = os.getenv("INFURA_PROJECT", "saske.xyz")
        
        # Настройка провайдера
        self.network = os.getenv("ETHEREUM_NETWORK", "sepolia")
        self.infura_url = f"https://{self.network}.infura.io/v3/{self.infura_api_key}"
        
        # Подключение к Ethereum
        self.connect()
        
        # Загрузка контрактов
        self.load_contracts()
        
        # Адрес кошелька и приватный ключ
        self.wallet_address = os.getenv("WALLET_ADDRESS")
        # ВАЖНО: приватный ключ никогда не должен быть захардкожен
        # Используем переменную окружения или другой безопасный способ хранения
        self._private_key = os.getenv("PRIVATE_KEY")
        
    def connect(self) -> bool:
        """
        Установка соединения с Ethereum
        
        Returns:
            bool: успешность подключения
        """
        try:
            self.w3 = Web3(Web3.HTTPProvider(self.infura_url))
            
            # Добавляем middleware для поддержки тестовых сетей (Sepolia, Goerli)
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            # Проверка соединения
            if not self.w3.is_connected():
                logger.error("Не удалось подключиться к Ethereum")
                return False
                
            # Получаем текущий блок для проверки
            current_block = self.w3.eth.block_number
            logger.info(f"Подключено к {self.network}, текущий блок: {current_block}")
            
            return True
        except Exception as e:
            logger.error(f"Ошибка подключения к Ethereum: {e}")
            return False
    
    def load_contracts(self) -> None:
        """
        Загрузка ABI контрактов
        """
        # Базовый путь к контрактам
        contract_dir = Path(__file__).parent / "contracts"
        
        # Загрузка EmotionalSBT
        try:
            # Поиск ABI-файла
            abi_path = contract_dir / "EmotionalSBT.abi"
            
            # Если файл не найден, используем шаблонное ABI
            if not abi_path.exists():
                logger.warning("ABI для EmotionalSBT не найден, используем шаблон")
                # В реальной системе здесь должна быть компиляция контракта с помощью solc
                # и генерация ABI, но для упрощения используем заглушку
                self.emotional_sbt_abi = []  # Заглушка для ABI
            else:
                # Загрузка ABI из файла
                with open(abi_path, 'r') as f:
                    self.emotional_sbt_abi = json.load(f)
            
            # Адрес контракта
            self.emotional_sbt_address = os.getenv('EMOTIONAL_SBT_ADDRESS')
            
            # Создаем экземпляр контракта если адрес указан
            if self.emotional_sbt_address:
                self.emotional_sbt_contract = self.w3.eth.contract(
                    address=self.emotional_sbt_address,
                    abi=self.emotional_sbt_abi
                )
                logger.info(f"Контракт EmotionalSBT загружен: {self.emotional_sbt_address}")
            else:
                logger.warning("Адрес контракта EmotionalSBT не указан")
                self.emotional_sbt_contract = None
        except Exception as e:
            logger.error(f"Ошибка загрузки контракта EmotionalSBT: {e}")
            self.emotional_sbt_contract = None
    
    def get_balance(self, address: Optional[str] = None) -> float:
        """
        Получение баланса в ETH
        
        Args:
            address: Адрес для проверки (если не указан, используется wallet_address)
            
        Returns:
            float: Баланс в ETH
        """
        if not address:
            address = self.wallet_address
            
        if not address:
            logger.error("Адрес не указан")
            return 0.0
            
        try:
            balance_wei = self.w3.eth.get_balance(address)
            balance_eth = self.w3.from_wei(balance_wei, 'ether')
            return float(balance_eth)
        except Exception as e:
            logger.error(f"Ошибка получения баланса: {e}")
            return 0.0
    
    def generate_zk_proof(self, emotion_data: Dict[str, Any], user_address: str) -> Dict[str, Any]:
        """
        Генерация Zero-Knowledge доказательства для эмоциональных данных
        
        Args:
            emotion_data: Данные об эмоциях
            user_address: Адрес пользователя
            
        Returns:
            Dict: ZK доказательство
        """
        # В реальной системе здесь должен быть настоящий ZK-протокол
        # Эта реализация является заглушкой для демонстрации концепции
        
        # Основная эмоция
        emotion = emotion_data.get("dominant_emotion", "neutral")
        intensity = min(100, int(emotion_data.get("dominant_score", 0.5) * 100))
        timestamp = int(time.time())
        
        # Создаем хэш из данных с солью для обеспечения приватности
        salt = os.urandom(16).hex()
        data_str = json.dumps({
            "emotion": emotion,
            "intensity": intensity,
            "timestamp": timestamp,
            "user": user_address,
            "salt": salt
        }, sort_keys=True)
        
        # Хэширование данных
        proof_hash = hashlib.sha256(data_str.encode()).hexdigest()
        
        # Формируем доказательство
        proof = {
            "emotion": emotion,
            "intensity": intensity,
            "timestamp": timestamp,
            "proof_hash": "0x" + proof_hash,
            "metadata": {
                "type": "emotional_state",
                "proof_version": "0.1",
                "is_verified": True
            }
        }
        
        logger.info(f"Сгенерировано ZK доказательство для {user_address}, эмоция: {emotion}")
        return proof
    
    def update_emotional_state(self, token_id: int, emotion_data: Dict[str, Any], 
                                user_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Обновление эмоционального состояния в контракте EmotionalSBT
        
        Args:
            token_id: ID токена
            emotion_data: Данные об эмоциях
            user_address: Адрес пользователя (если не указан, используется wallet_address)
            
        Returns:
            Dict: Результат операции
        """
        if not self.emotional_sbt_contract:
            logger.error("Контракт EmotionalSBT не инициализирован")
            return {"success": False, "error": "Контракт не инициализирован"}
            
        if not user_address:
            user_address = self.wallet_address
            
        if not user_address:
            logger.error("Адрес пользователя не указан")
            return {"success": False, "error": "Адрес пользователя не указан"}
            
        try:
            # Генерируем ZK доказательство
            proof = self.generate_zk_proof(emotion_data, user_address)
            
            # В реальной системе здесь должна быть отправка транзакции в блокчейн
            # Для демонстрации возвращаем фиктивный результат
            
            # Имитация вызова контракта (в реальной системе будет вызов updateEmotionalState)
            tx_result = {
                "success": True,
                "tx_hash": "0x" + os.urandom(32).hex(),
                "token_id": token_id,
                "emotion": proof["emotion"],
                "intensity": proof["intensity"],
                "timestamp": proof["timestamp"],
                "proof_hash": proof["proof_hash"]
            }
            
            logger.info(f"Эмоциональное состояние обновлено: {tx_result['emotion']}")
            return tx_result
        except Exception as e:
            logger.error(f"Ошибка обновления эмоционального состояния: {e}")
            return {"success": False, "error": str(e)}
    
    def get_emotional_state(self, token_id: int) -> Dict[str, Any]:
        """
        Получение текущего эмоционального состояния из контракта
        
        Args:
            token_id: ID токена
            
        Returns:
            Dict: Данные об эмоциональном состоянии
        """
        if not self.emotional_sbt_contract:
            logger.error("Контракт EmotionalSBT не инициализирован")
            return {"success": False, "error": "Контракт не инициализирован"}
            
        try:
            # В реальной системе здесь должен быть вызов контракта
            # Для демонстрации возвращаем фиктивные данные
            
            # Возможные эмоции
            emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
            
            # Имитация данных
            state = {
                "success": True,
                "token_id": token_id,
                "emotion": random.choice(emotions),
                "intensity": random.randint(30, 90),
                "timestamp": int(time.time()) - random.randint(0, 3600),
                "proof_hash": "0x" + os.urandom(32).hex(),
                "verified": True
            }
            
            logger.info(f"Получено эмоциональное состояние для токена {token_id}")
            return state
        except Exception as e:
            logger.error(f"Ошибка получения эмоционального состояния: {e}")
            return {"success": False, "error": str(e)}
    
    def close(self) -> None:
        """
        Закрытие соединения с Ethereum
        """
        logger.info("Web3 соединение закрыто")


class EmotionalZKProof:
    """
    Класс для работы с Zero-Knowledge доказательствами эмоциональных данных
    Используется для защиты приватности пользователей в SASOK
    """
    
    def __init__(self, salt: Optional[str] = None):
        """
        Инициализация
        
        Args:
            salt: Соль для хэширования (если не указана, генерируется автоматически)
        """
        self.salt = salt or os.urandom(16).hex()
        
    def generate_proof(self, emotion_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Генерация доказательства для эмоциональных данных без раскрытия самих данных
        
        Args:
            emotion_data: Данные об эмоциях
            user_id: Идентификатор пользователя
            
        Returns:
            Dict: Доказательство
        """
        # Извлекаем ключевую информацию
        emotion = emotion_data.get("dominant_emotion", "neutral")
        score = emotion_data.get("dominant_score", 0.5)
        timestamp = emotion_data.get("timestamp", time.time())
        
        # Хэшируем полные данные с солью
        data_str = json.dumps(emotion_data, sort_keys=True)
        data_hash = hashlib.sha256((data_str + self.salt).encode()).hexdigest()
        
        # Создаем второй хэш для проверки без раскрытия данных
        verification_str = json.dumps({
            "emotion": emotion,
            "score_range": int(score * 10) / 10,  # Округляем до 0.1
            "timestamp_hour": int(timestamp / 3600),
            "user_id": user_id
        }, sort_keys=True)
        verification_hash = hashlib.sha256((verification_str + self.salt).encode()).hexdigest()
        
        # Формируем доказательство
        proof = {
            "data_hash": data_hash,
            "verification_hash": verification_hash,
            "emotion_type": emotion,
            "intensity_range": str(int(score * 10) / 10),
            "timestamp": int(timestamp),
            "user_id_hash": hashlib.sha256(user_id.encode()).hexdigest(),
            "proof_id": os.urandom(8).hex()
        }
        
        return proof
    
    def verify_proof(self, proof: Dict[str, Any], original_data: Dict[str, Any], 
                    user_id: str) -> bool:
        """
        Проверка соответствия доказательства оригинальным данным
        
        Args:
            proof: Доказательство
            original_data: Оригинальные данные
            user_id: Идентификатор пользователя
            
        Returns:
            bool: Результат проверки
        """
        # Хэшируем оригинальные данные
        data_str = json.dumps(original_data, sort_keys=True)
        expected_data_hash = hashlib.sha256((data_str + self.salt).encode()).hexdigest()
        
        # Извлекаем информацию для верификации
        emotion = original_data.get("dominant_emotion", "neutral")
        score = original_data.get("dominant_score", 0.5)
        timestamp = original_data.get("timestamp", time.time())
        
        # Создаем второй хэш для проверки
        verification_str = json.dumps({
            "emotion": emotion,
            "score_range": int(score * 10) / 10,
            "timestamp_hour": int(timestamp / 3600),
            "user_id": user_id
        }, sort_keys=True)
        expected_verification_hash = hashlib.sha256((verification_str + self.salt).encode()).hexdigest()
        
        # Проверяем совпадение хэшей
        data_match = proof["data_hash"] == expected_data_hash
        verification_match = proof["verification_hash"] == expected_verification_hash
        
        return data_match and verification_match
