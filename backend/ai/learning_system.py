import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import os

class LearningSystem:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.interaction_history = []
        self.model_performance = {}
        self.training_data = {}
        self.initialize_system()

    def initialize_system(self):
        """Initialize the learning system"""
        try:
            # Load existing training data if available
            if os.path.exists('data/training_data.json'):
                with open('data/training_data.json', 'r') as f:
                    self.training_data = json.load(f)
            
            # Initialize performance tracking
            self.model_performance = {
                'emotion_detection': {
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': []
                },
                'nlp_processing': {
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': []
                }
            }

            self.logger.info("Learning system initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing learning system: {str(e)}")
            raise

    async def log_interaction(self, interaction_data: Dict) -> Dict:
        """Log user interaction for learning"""
        try:
            interaction = {
                'timestamp': datetime.utcnow().isoformat(),
                'data': interaction_data,
                'feedback': interaction_data.get('feedback'),
                'performance': interaction_data.get('performance', {})
            }
            
            self.interaction_history.append(interaction)
            
            # Update training data if feedback is provided
            if interaction_data.get('feedback'):
                await self.update_training_data(interaction_data)

            return {
                'success': True,
                'interaction_id': len(self.interaction_history)
            }

        except Exception as e:
            self.logger.error(f"Error logging interaction: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    async def update_training_data(self, interaction_data: Dict):
        """Update training data based on user interaction"""
        try:
            data_type = interaction_data.get('type')
            if not data_type:
                return

            if data_type not in self.training_data:
                self.training_data[data_type] = []

            training_entry = {
                'input': interaction_data.get('input'),
                'output': interaction_data.get('output'),
                'feedback': interaction_data.get('feedback'),
                'timestamp': datetime.utcnow().isoformat()
            }

            self.training_data[data_type].append(training_entry)
            
            # Save updated training data
            self._save_training_data()

        except Exception as e:
            self.logger.error(f"Error updating training data: {str(e)}")
            raise

    async def train_models(self, model_type: str) -> Dict:
        """Train or update models based on collected data"""
        try:
            if model_type not in self.training_data:
                return {
                    'success': False,
                    'error': 'No training data available for this model type'
                }

            training_data = self.training_data[model_type]
            if not training_data:
                return {
                    'success': False,
                    'error': 'Empty training dataset'
                }

            # Prepare data for training
            X, y = self._prepare_training_data(training_data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train model (placeholder for actual training logic)
            training_result = self._train_model(
                model_type, X_train, y_train, X_test, y_test
            )

            return {
                'success': True,
                'model_type': model_type,
                'metrics': training_result
            }

        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def _prepare_training_data(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model training"""
        # Placeholder for actual data preparation logic
        X = np.array([entry['input'] for entry in data])
        y = np.array([entry['output'] for entry in data])
        return X, y

    def _train_model(self, 
                    model_type: str, 
                    X_train: np.ndarray, 
                    y_train: np.ndarray,
                    X_test: np.ndarray,
                    y_test: np.ndarray) -> Dict:
        """Train a specific model type"""
        # Placeholder for actual model training logic
        return {
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.82,
            'f1': 0.84
        }

    def _save_training_data(self):
        """Save training data to disk"""
        try:
            os.makedirs('data', exist_ok=True)
            with open('data/training_data.json', 'w') as f:
                json.dump(self.training_data, f)
        except Exception as e:
            self.logger.error(f"Error saving training data: {str(e)}")
            raise

    async def analyze_performance(self, model_type: str) -> Dict:
        """Analyze model performance over time"""
        try:
            if model_type not in self.model_performance:
                return {
                    'success': False,
                    'error': 'Invalid model type'
                }

            performance = self.model_performance[model_type]
            
            return {
                'success': True,
                'model_type': model_type,
                'metrics': {
                    'accuracy': {
                        'current': performance['accuracy'][-1] if performance['accuracy'] else None,
                        'trend': self._calculate_trend(performance['accuracy'])
                    },
                    'precision': {
                        'current': performance['precision'][-1] if performance['precision'] else None,
                        'trend': self._calculate_trend(performance['precision'])
                    },
                    'recall': {
                        'current': performance['recall'][-1] if performance['recall'] else None,
                        'trend': self._calculate_trend(performance['recall'])
                    },
                    'f1': {
                        'current': performance['f1'][-1] if performance['f1'] else None,
                        'trend': self._calculate_trend(performance['f1'])
                    }
                }
            }

        except Exception as e:
            self.logger.error(f"Error analyzing performance: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def _calculate_trend(self, metrics: List[float]) -> str:
        """Calculate trend direction from metrics history"""
        if not metrics or len(metrics) < 2:
            return 'stable'
        
        recent = metrics[-5:] if len(metrics) >= 5 else metrics
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        return 'stable'

    async def get_learning_stats(self) -> Dict:
        """Get statistics about the learning system"""
        return {
            'total_interactions': len(self.interaction_history),
            'training_data_size': {
                model_type: len(data)
                for model_type, data in self.training_data.items()
            },
            'model_performance': {
                model_type: await self.analyze_performance(model_type)
                for model_type in self.model_performance.keys()
            },
            'last_updated': datetime.utcnow().isoformat()
        }
