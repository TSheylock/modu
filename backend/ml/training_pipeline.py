import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import optuna
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support
import json
import os
from pathlib import Path
import shap
import lime
import lime.lime_text

class EmotionDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class MultiModalEmotionModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class TrainingPipeline:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.best_params = None
        self.training_history = []

    def prepare_data(self, 
                    features: np.ndarray, 
                    labels: np.ndarray,
                    val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """Prepare data loaders for training"""
        try:
            # Split data
            indices = np.random.permutation(len(features))
            split_idx = int(len(features) * (1 - val_split))
            
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
            
            # Create datasets
            train_dataset = EmotionDataset(
                features[train_indices],
                labels[train_indices]
            )
            val_dataset = EmotionDataset(
                features[val_indices],
                labels[val_indices]
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size']
            )
            
            return train_loader, val_loader

        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function for hyperparameter optimization"""
        try:
            # Define hyperparameters to optimize
            hidden_dims = [
                trial.suggest_int(f"hidden_dim_{i}", 32, 512)
                for i in range(trial.suggest_int("n_layers", 1, 3))
            ]
            learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-2)
            
            # Create model with trial parameters
            self.model = MultiModalEmotionModel(
                self.config['input_dim'],
                hidden_dims,
                self.config['num_classes']
            ).to(self.device)
            
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate
            )
            
            # Train for a few epochs
            val_scores = []
            for epoch in range(self.config['num_epochs']):
                train_loss = self._train_epoch(self.train_loader)
                val_loss, val_metrics = self._validate_epoch(self.val_loader)
                val_scores.append(val_metrics['f1'])
                
                # Report intermediate value
                trial.report(val_metrics['f1'], epoch)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return max(val_scores)

        except Exception as e:
            self.logger.error(f"Error in optimization objective: {str(e)}")
            raise

    def train(self, 
              features: np.ndarray,
              labels: np.ndarray,
              n_trials: int = 100) -> Dict:
        """Train the model with hyperparameter optimization"""
        try:
            # Prepare data
            self.train_loader, self.val_loader = self.prepare_data(
                features, labels
            )
            
            # Create study for hyperparameter optimization
            study = optuna.create_study(
                direction="maximize",
                pruner=optuna.pruners.MedianPruner()
            )
            study.optimize(self.objective, n_trials=n_trials)
            
            # Get best parameters
            self.best_params = study.best_params
            
            # Train final model with best parameters
            self.model = MultiModalEmotionModel(
                self.config['input_dim'],
                [self.best_params[f"hidden_dim_{i}"] 
                 for i in range(self.best_params["n_layers"])],
                self.config['num_classes']
            ).to(self.device)
            
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.best_params["lr"]
            )
            
            # Final training
            for epoch in range(self.config['num_epochs']):
                train_loss = self._train_epoch(self.train_loader)
                val_loss, val_metrics = self._validate_epoch(self.val_loader)
                
                self.training_history.append({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **val_metrics
                })
            
            return {
                "success": True,
                "best_params": self.best_params,
                "training_history": self.training_history,
                "final_metrics": self.training_history[-1]
            }

        except Exception as e:
            self.logger.error(f"Error in training: {str(e)}")
            return {"success": False, "error": str(e)}

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for features, labels in train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)

    def _validate_epoch(self, 
                       val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels,
            all_preds,
            average='weighted'
        )
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        return total_loss / len(val_loader), metrics

    def explain_prediction(self, 
                         features: np.ndarray,
                         method: str = 'shap') -> Dict:
        """Generate model explanations"""
        try:
            if method == 'shap':
                explainer = shap.DeepExplainer(
                    self.model,
                    torch.FloatTensor(features[:100]).to(self.device)
                )
                shap_values = explainer.shap_values(
                    torch.FloatTensor(features).to(self.device)
                )
                
                return {
                    "success": True,
                    "explanations": {
                        "shap_values": shap_values,
                        "feature_importance": np.abs(shap_values).mean(0)
                    }
                }
            
            elif method == 'lime':
                explainer = lime.lime_text.LimeTextExplainer()
                exp = explainer.explain_instance(
                    features,
                    self.model.predict_proba,
                    num_features=10
                )
                
                return {
                    "success": True,
                    "explanations": {
                        "feature_weights": exp.as_list(),
                        "local_prediction": exp.local_pred
                    }
                }
            
            else:
                raise ValueError(f"Unsupported explanation method: {method}")

        except Exception as e:
            self.logger.error(f"Error generating explanations: {str(e)}")
            return {"success": False, "error": str(e)}

    def save_model(self, path: str) -> Dict:
        """Save model and training artifacts"""
        try:
            save_dict = {
                "model_state": self.model.state_dict(),
                "best_params": self.best_params,
                "training_history": self.training_history,
                "config": self.config
            }
            
            torch.save(save_dict, path)
            
            return {
                "success": True,
                "path": path,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return {"success": False, "error": str(e)}

    def load_model(self, path: str) -> Dict:
        """Load model and training artifacts"""
        try:
            save_dict = torch.load(path)
            
            # Recreate model with saved config
            self.config = save_dict['config']
            self.model = MultiModalEmotionModel(
                self.config['input_dim'],
                [save_dict['best_params'][f"hidden_dim_{i}"] 
                 for i in range(save_dict['best_params']["n_layers"])],
                self.config['num_classes']
            ).to(self.device)
            
            self.model.load_state_dict(save_dict['model_state'])
            self.best_params = save_dict['best_params']
            self.training_history = save_dict['training_history']
            
            return {
                "success": True,
                "config": self.config,
                "best_params": self.best_params,
                "training_history": self.training_history
            }

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return {"success": False, "error": str(e)}
