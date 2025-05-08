import torch
from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor
import torchaudio
import cv2
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
from datetime import datetime
import json
import os
from pathlib import Path

class MultiModalPipeline:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.tokenizers = {}
        self.feature_extractors = {}
        self.initialize_models()

    def initialize_models(self):
        """Initialize all required models and processors"""
        try:
            # Text models (BERT-based emotion detection)
            self.tokenizers['text'] = AutoTokenizer.from_pretrained(
                'j-hartmann/emotion-english-distilroberta-base'
            )
            self.models['text'] = AutoModel.from_pretrained(
                'j-hartmann/emotion-english-distilroberta-base'
            )

            # Audio models (Wav2Vec based)
            self.models['audio'] = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
            
            # Vision models
            self.feature_extractors['vision'] = AutoFeatureExtractor.from_pretrained(
                'microsoft/resnet-50'
            )
            self.models['vision'] = AutoModel.from_pretrained(
                'microsoft/resnet-50'
            )

            self.logger.info("All models initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise

    async def process_text(self, text: str) -> Dict:
        """Process text input for emotion analysis"""
        try:
            # Tokenize and encode text
            inputs = self.tokenizers['text'](
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            # Get model outputs
            outputs = self.models['text'](**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)

            return {
                "success": True,
                "embeddings": embeddings.detach().numpy(),
                "text_length": len(text.split()),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error processing text: {str(e)}")
            return {"success": False, "error": str(e)}

    async def process_audio(self, audio_path: str) -> Dict:
        """Process audio input for emotion analysis"""
        try:
            # Load and preprocess audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, 16000
                )
                waveform = resampler(waveform)

            # Extract features
            with torch.no_grad():
                features, _ = self.models['audio'](waveform)

            return {
                "success": True,
                "features": features.numpy(),
                "duration": waveform.shape[1] / 16000,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error processing audio: {str(e)}")
            return {"success": False, "error": str(e)}

    async def process_video(self, video_path: str) -> Dict:
        """Process video input for emotion analysis"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5
                )

                # Process each face
                for (x, y, w, h) in faces:
                    face = frame[y:y+h, x:x+w]
                    # Prepare for vision model
                    inputs = self.feature_extractors['vision'](
                        face, return_tensors="pt"
                    )
                    features = self.models['vision'](**inputs)
                    frames.append(features.pooler_output.detach().numpy())

            cap.release()

            return {
                "success": True,
                "frame_features": np.array(frames),
                "face_count": len(frames),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}")
            return {"success": False, "error": str(e)}

    async def process_multimodal(self, 
                               data: Dict[str, Union[str, bytes]]) -> Dict:
        """Process multiple modalities together"""
        try:
            results = {}
            
            # Process each modality
            if 'text' in data:
                results['text'] = await self.process_text(data['text'])
            
            if 'audio' in data:
                results['audio'] = await self.process_audio(data['audio'])
            
            if 'video' in data:
                results['video'] = await self.process_video(data['video'])

            # Combine features
            combined_features = self.combine_features(results)

            return {
                "success": True,
                "individual_results": results,
                "combined_features": combined_features,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error in multimodal processing: {str(e)}")
            return {"success": False, "error": str(e)}

    def combine_features(self, results: Dict) -> np.ndarray:
        """Combine features from different modalities"""
        try:
            features = []
            
            # Add text features if available
            if 'text' in results and results['text']['success']:
                features.append(results['text']['embeddings'])

            # Add audio features if available
            if 'audio' in results and results['audio']['success']:
                features.append(results['audio']['features'])

            # Add video features if available
            if 'video' in results and results['video']['success']:
                # Average across frames
                features.append(results['video']['frame_features'].mean(axis=0))

            # Concatenate all features
            if features:
                return np.concatenate(features, axis=-1)
            return np.array([])

        except Exception as e:
            self.logger.error(f"Error combining features: {str(e)}")
            return np.array([])

    async def save_to_ipfs(self, 
                          data: Dict,
                          web3_service: any) -> Dict:
        """Save processed data to IPFS"""
        try:
            # Prepare metadata
            metadata = {
                "timestamp": datetime.utcnow().isoformat(),
                "modalities": list(data.keys()),
                "feature_dimensions": {
                    k: v['features'].shape if 'features' in v else None
                    for k, v in data.items()
                    if v.get('success', False)
                }
            }

            # Save to IPFS (placeholder for actual IPFS integration)
            ipfs_hash = "QmHash..."  # Replace with actual IPFS upload

            return {
                "success": True,
                "ipfs_hash": ipfs_hash,
                "metadata": metadata
            }

        except Exception as e:
            self.logger.error(f"Error saving to IPFS: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        return {
            "text_model": "emotion-english-distilroberta-base",
            "audio_model": "wav2vec2-base",
            "vision_model": "resnet-50",
            "status": {
                k: v is not None 
                for k, v in self.models.items()
            }
        }
