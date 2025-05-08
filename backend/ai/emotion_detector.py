import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import logging
from typing import Dict, Optional, Tuple
import base64
import io
from PIL import Image
import uuid
from requests import post
import transformers
from transformers import pipeline

class EmotionDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = None
        self.initialize_model()
        self.empathy_styles = ['sympathetic', 'encouraging', 'mirroring']  
        self.llm_pipeline = pipeline('text-generation', model='meta-llama/Meta-Llama-3-8B', device=0)  

    def initialize_model(self):
        try:
            self.model = load_model('models/emotion_model.h5')
            self.logger.info("Emotion detection model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading emotion detection model: {str(e)}")
            raise

    async def process_image(self, image_data: str) -> Dict:
        result = await self._process_image_internal(image_data)  
        if result.get("success", False):
            for emotion_result in result['results']:
                await self.add_emotion_to_graph(emotion_result)
                context = f"User emotion detected: {emotion_result['emotion']} with confidence {emotion_result['confidence']}"
                response = self.generate_response(context, style='sympathetic')  
                emotion_result['llm_response'] = response  
        return result

    async def _process_image_internal(self, image_data: str) -> Dict:
        try:
            image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            faces = self.detect_faces(image_cv)
            if not faces:
                return {"success": False, "error": "No faces detected in the image"}
            results = []
            for face in faces:
                emotion_data = self.analyze_face(image_cv, face)
                results.append(emotion_data)
            return {"success": True, "faces_detected": len(faces), "results": results}
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return {"success": False, "error": str(e)}

    async def process_video_frame(self, frame: np.ndarray) -> Dict:
        result = self._process_video_frame_internal(frame)
        if result.get("success", False):
            for emotion_result in result['results']:
                await self.add_emotion_to_graph(emotion_result)
                context = f"User emotion detected: {emotion_result['emotion']} with confidence {emotion_result['confidence']}"
                response = self.generate_response(context, style='sympathetic')
                emotion_result['llm_response'] = response
        return result

    def _process_video_frame_internal(self, frame: np.ndarray) -> Dict:
        try:
            faces = self.detect_faces(frame)
            if not faces:
                return {"success": False, "error": "No faces detected in frame"}
            results = []
            for face in faces:
                emotion_data = self.analyze_face(frame, face)
                results.append(emotion_data)
            return {"success": True, "faces_detected": len(faces), "results": results}
        except Exception as e:
            self.logger.error(f"Error processing video frame: {str(e)}")
            return {"success": False, "error": str(e)}

    async def add_emotion_to_graph(self, emotion_data: Dict):
        try:
            response = post("http://localhost:3000/api/memory/add_node", json={
                "id": uuid.uuid4().hex,
                "type": "emotion",
                "label": emotion_data['emotion'],
                "score": emotion_data['confidence']
            })
            if response.status_code != 200:
                self.logger.error(f"Failed to add node to graph: {response.status_code} - {response.text}")
            else:
                self.logger.info("Emotion node added successfully")
        except Exception as e:
            self.logger.error(f"Error adding emotion to graph: {str(e)}")

    def generate_response(self, context: str, style: str = 'sympathetic') -> str:
        if style not in self.empathy_styles:
            style = 'sympathetic'  
            self.logger.warning(f"Invalid empathy style provided. Defaulting to 'sympathetic'.")
        
        prompt = f"Respond in a {style} manner: {context}"
        try:
            response = self.llm_pipeline(prompt, max_length=100, num_return_sequences=1)
            return response[0]['generated_text']
        except Exception as e:
            self.logger.error(f"Error generating LLM response: {str(e)}")
            return "Unable to generate response at this time."

    def detect_faces(self, image: np.ndarray) -> list:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces

    def analyze_face(self, image: np.ndarray, face: Tuple) -> Dict:
        try:
            x, y, w, h = face
            roi_gray = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = self.model.predict(roi)[0]
            emotion_probabilities = {
                emotion: float(prob) 
                for emotion, prob in zip(self.emotions, prediction)
            }
            
            max_emotion = max(emotion_probabilities.items(), key=lambda x: x[1])

            return {
                "emotion": max_emotion[0],
                "confidence": max_emotion[1],
                "all_emotions": emotion_probabilities,
                "face_location": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                }
            }

        except Exception as e:
            self.logger.error(f"Error analyzing face: {str(e)}")
            return {
                "error": str(e)
            }

    def draw_results(self, image: np.ndarray, results: Dict) -> np.ndarray:
        if not results["success"]:
            return image

        for result in results["results"]:
            if "error" in result:
                continue

            face_loc = result["face_location"]
            emotion = result["emotion"]
            confidence = result["confidence"]

            cv2.rectangle(
                image,
                (face_loc["x"], face_loc["y"]),
                (face_loc["x"] + face_loc["width"], face_loc["y"] + face_loc["height"]),
                (0, 255, 0),
                2
            )

            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(
                image,
                label,
                (face_loc["x"], face_loc["y"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                2
            )

        return image

    def get_model_info(self) -> Dict:
        return {
            "emotions_supported": self.emotions,
            "model_loaded": self.model is not None,
            "input_shape": self.model.input_shape if self.model else None,
            "version": "1.0.0"
        }
