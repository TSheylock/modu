# ASR Service Adapter for SASOK
import requests
import os
from typing import Dict, Any, Optional

class ASRServiceAdapter:
    """
    Adapter for NeMo ASR service following SASOK architecture principles.
    Processes audio files to extract speech transcription while preserving
    privacy and maintaining emotional context.
    """
    
    def __init__(self, service_url: str = "http://localhost:5000"):
        self.service_url = service_url
        self.session = requests.Session()
        self.model_initialized = False
    
    def initialize_model(self, model_name: str = "stt_en_conformer_ctc_large") -> bool:
        """Initialize the ASR model"""
        try:
            response = self.session.post(
                f"{self.service_url}/initialize_model",
                data={"model_names_select": model_name, "use_gpu_ckbx": "on"}
            )
            self.model_initialized = response.status_code == 200
            return self.model_initialized
        except Exception as e:
            print(f"SASOK_DOUBT: ASR model initialization failed: {e}")
            return False
    
    def transcribe_audio(self, audio_path: str) -> Optional[str]:
        """Transcribe audio file to text for emotion analysis"""
        if not self.model_initialized:
            if not self.initialize_model():
                return None
                
        files = {'file': open(audio_path, 'rb')}
        
        # Upload audio
        upload_response = self.session.post(
            f"{self.service_url}/upload_audio_files",
            files=files
        )
        
        if upload_response.status_code != 200:
            return None
            
        # Transcribe
        transcribe_response = self.session.post(
            f"{self.service_url}/transcribe"
        )
        
        if transcribe_response.status_code == 200:
            # Extract transcription text from response
            # (simplified - actual implementation would parse the HTML response)
            return transcribe_response.text
        
        return None

# Integration with your existing emotion analysis system
def process_audio_for_emotion(audio_path: str) -> Dict[str, Any]:
    """
    Process audio file for emotion analysis, extracting both
    transcription and audio emotional features
    """
    # Get text transcription from ASR
    asr_adapter = ASRServiceAdapter()
    transcription = asr_adapter.transcribe_audio(audio_path)
    
    if not transcription:
        return {"status": "error", "emotion": None, "text": None}
    
    # Use your existing text emotion analysis
    text_emotion = analyze_text_emotion(transcription)
    
    # Use your existing audio emotion analysis on the same file
    audio_emotion = analyze_audio_emotion(audio_path)
    
    # Combine using your multimodal approach
    combined_emotion = combine_emotion_signals({
        "text": text_emotion,
        "audio": audio_emotion
    })
    
    return {
        "status": "success",
        "emotion": combined_emotion,
        "text": transcription
    }# Add to backend/ai/emotion_detector.py
    
    import requests
    import os
    import tempfile
    import base64
    import wave
    import numpy as np
    import librosa
    
    async def process_audio(self, audio_data: str) -> Dict:
        """
        Process audio data for both speech recognition and emotion analysis.
        Integrates with NeMo ASR for transcription and performs audio emotion analysis.
        
        Args:
            audio_data: Base64 encoded audio data
        
        Returns:
            Dict containing transcription, emotion analysis, and metadata
        """
        try:
            # Decode audio data
            audio_bytes = base64.b64decode(audio_data.split(',')[1] if ',' in audio_data else audio_data)
            
            # Save to temporary WAV file (16kHz mono required for ASR)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                with wave.open(temp_path, 'wb') as wf:
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(16000)  # 16kHz
                    wf.writeframes(audio_bytes)
            
            # Get transcription from NeMo ASR service
            transcription = await self._get_asr_transcription(temp_path)
            
            # Analyze audio for emotional content
            emotion_data = await self._analyze_audio_emotion(temp_path)
            
            # Analyze text from transcription
            text_emotion = await self._analyze_text_emotion(transcription)
            
            # Combine emotion signals for more accurate analysis
            combined_emotion = self._combine_emotion_signals({
                "audio": emotion_data,
                "text": text_emotion
            })
            
            # Add to emotion graph
            await self.add_emotion_to_graph(combined_emotion)
            
            # Generate appropriate response
            context = f"User said: '{transcription}' with emotion {combined_emotion['emotion']}"
            response = self.generate_response(context, style='mirroring')
            
            # Clean up temp file
            os.unlink(temp_path)
            
            return {
                "success": True,
                "transcription": transcription,
                "emotion": combined_emotion,
                "response": response
            }
        except Exception as e:
            self.logger.error(f"Error processing audio: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _get_asr_transcription(self, audio_path: str) -> str:
        """
        Get transcription from NeMo ASR service
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Transcribed text
        """
        try:
            # First initialize the model if not already done
            session = requests.Session()
            
            # Step 1: Initialize model
            model_response = session.post(
                "http://localhost:5000/initialize_model",
                data={
                    "model_names_select": "stt_en_conformer_ctc_large",  # Default English model
                    "use_gpu_ckbx": "on"
                }
            )
            
            if model_response.status_code != 200:
                raise Exception(f"Failed to initialize ASR model: {model_response.status_code}")
            
            # Step 2: Upload audio file
            files = {'file': open(audio_path, 'rb')}
            upload_response = session.post(
                "http://localhost:5000/upload_audio_files",
                files=files
            )
            
            if upload_response.status_code != 200:
                raise Exception(f"Failed to upload audio: {upload_response.status_code}")
            
            # Step 3: Transcribe
            transcribe_response = session.post(
                "http://localhost:5000/transcribe"
            )
            
            if transcribe_response.status_code != 200:
                raise Exception(f"Failed to transcribe: {transcribe_response.status_code}")
            
            # Extract transcription text from response HTML
            # This is a simplified approach and may need adjustment based on actual response format
            response_text = transcribe_response.text
            # Basic extraction - in production would use proper HTML parsing
            start_marker = '<div class="transcription">'
            end_marker = '</div>'
            
            if start_marker in response_text and end_marker in response_text:
                start_idx = response_text.find(start_marker) + len(start_marker)
                end_idx = response_text.find(end_marker, start_idx)
                transcription = response_text[start_idx:end_idx].strip()
                return transcription
            
            return "Transcription unavailable"
                
        except Exception as e:
            self.logger.error(f"Error getting ASR transcription: {str(e)}")
            return "Transcription failed"
    
    async def _analyze_audio_emotion(self, audio_path: str) -> Dict:
        """
        Analyze audio for emotional content using wav2vec or similar model
        This implements the audio emotion analysis mentioned in your memories
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Dict with emotion analysis results
        """
        try:
            # Load audio file with librosa (common for audio processing)
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Extract audio features
            # This is a placeholder for actual audio emotion analysis
            # In production, you would use the wav2vec model mentioned in your memories
            
            # Mock result for demonstration
            emotion_probs = {
                "angry": 0.05,
                "disgust": 0.02,
                "fear": 0.03,
                "happy": 0.65,
                "sad": 0.10,
                "surprise": 0.05,
                "neutral": 0.10
            }
            
            max_emotion = max(emotion_probs.items(), key=lambda x: x[1])
            
            return {
                "emotion": max_emotion[0],
                "confidence": max_emotion[1],
                "all_emotions": emotion_probs,
                "source": "audio"
            }
        except Exception as e:
            self.logger.error(f"Error analyzing audio emotion: {str(e)}")
            return {
                "emotion": "neutral",
                "confidence": 1.0,
                "all_emotions": {"neutral": 1.0},
                "source": "audio",
                "error": str(e)
            }
    
    async def _analyze_text_emotion(self, text: str) -> Dict:
        """
        Analyze text for emotional content using DistilBERT or similar model
        This implements the text emotion analysis mentioned in your memories
        
        Args:
            text: Transcribed text to analyze
        
        Returns:
            Dict with emotion analysis results
        """
        try:
            # In production, you would use the DistilBERT model mentioned in your memories
            # This is a placeholder for actual text emotion analysis
            
            # Mock result for demonstration
            emotion_probs = {
                "angry": 0.02,
                "disgust": 0.01,
                "fear": 0.02,
                "happy": 0.55,
                "sad": 0.05,
                "surprise": 0.05,
                "neutral": 0.30
            }
            
            max_emotion = max(emotion_probs.items(), key=lambda x: x[1])
            
            return {
                "emotion": max_emotion[0],
                "confidence": max_emotion[1],
                "all_emotions": emotion_probs,
                "source": "text"
            }
        except Exception as e:
            self.logger.error(f"Error analyzing text emotion: {str(e)}")
            return {
                "emotion": "neutral",
                "confidence": 1.0,
                "all_emotions": {"neutral": 1.0},
                "source": "text",
                "error": str(e)
            }
    
    def _combine_emotion_signals(self, emotions: Dict) -> Dict:
        """
        Combine emotion signals from different modalities
        
        Args:
            emotions: Dict with emotion analysis from different sources
        
        Returns:
            Combined emotion result
        """
        # Simple weighted average approach
        # In production, you might use a more sophisticated fusion model
        weights = {
            "audio": 0.4,
            "text": 0.6
        }
        
        combined_probs = {emotion: 0.0 for emotion in self.emotions}
        
        for source, emotion_data in emotions.items():
            if source in weights and "all_emotions" in emotion_data:
                source_weight = weights[source]
                for emotion, prob in emotion_data["all_emotions"].items():
                    if emotion in combined_probs:
                        combined_probs[emotion] += prob * source_weight
        
        max_emotion = max(combined_probs.items(), key=lambda x: x[1])
        
        return {
            "emotion": max_emotion[0],
            "confidence": max_emotion[1],
            "all_emotions": combined_probs,
            "source": "multimodal"
        }