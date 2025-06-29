from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, List, Optional
import logging
import numpy as np
from datetime import datetime

class NLPProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tokenizer = None
        self.model = None
        self.sentiment_analyzer = None
        self.intent_classifier = None
        self.response_generator = None
        self.initialize_models()

    def initialize_models(self):
        """Initialize all NLP models"""
        try:
            # Initialize sentiment analysis
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )

            # Initialize intent classification
            self.intent_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )

            # Initialize response generation
            self.response_generator = pipeline(
                "text-generation",
                model="gpt2"
            )

            self.logger.info("NLP models initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing NLP models: {str(e)}")
            raise

    async def process_text(self, text: str, context: Optional[List[Dict]] = None) -> Dict:
        """
        Process text input with sentiment analysis, intent classification,
        and response generation
        """
        try:
            # Analyze sentiment
            sentiment = await self.analyze_sentiment(text)
            
            # Classify intent
            intent = await self.classify_intent(text)
            
            # Generate response
            response = await self.generate_response(text, context)

            return {
                "success": True,
                "sentiment": sentiment,
                "intent": intent,
                "response": response,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error processing text: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment in text"""
        try:
            result = self.sentiment_analyzer(text)[0]
            return {
                "label": result["label"],
                "score": float(result["score"])
            }
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return {
                "label": "unknown",
                "score": 0.0
            }

    async def classify_intent(self, text: str) -> Dict:
        """Classify user intent"""
        try:
            candidate_labels = [
                "question",
                "request",
                "statement",
                "command",
                "greeting",
                "farewell"
            ]
            
            result = self.intent_classifier(
                text,
                candidate_labels,
                multi_label=False
            )

            return {
                "intent": result["labels"][0],
                "confidence": float(result["scores"][0]),
                "all_intents": [
                    {"label": label, "score": float(score)}
                    for label, score in zip(result["labels"], result["scores"])
                ]
            }
        except Exception as e:
            self.logger.error(f"Error in intent classification: {str(e)}")
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "all_intents": []
            }

    async def generate_response(self, 
                              text: str, 
                              context: Optional[List[Dict]] = None) -> Dict:
        """Generate response based on input and context"""
        try:
            # Prepare context
            context_text = ""
            if context:
                context_text = " ".join([
                    f"{msg['role']}: {msg['content']}"
                    for msg in context[-3:]  # Use last 3 messages for context
                ])

            # Generate response
            response = self.response_generator(
                f"{context_text} User: {text}",
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9
            )[0]["generated_text"]

            return {
                "text": response,
                "confidence": 0.85,  # Placeholder confidence score
                "context_used": bool(context)
            }

        except Exception as e:
            self.logger.error(f"Error in response generation: {str(e)}")
            return {
                "text": "I apologize, but I'm having trouble generating a response.",
                "confidence": 0.0,
                "context_used": False
            }

    async def extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities from text"""
        try:
            ner_pipeline = pipeline("ner", grouped_entities=True)
            entities = ner_pipeline(text)
            
            return [
                {
                    "text": entity["word"],
                    "type": entity["entity_group"],
                    "score": float(entity["score"]),
                    "start": entity["start"],
                    "end": entity["end"]
                }
                for entity in entities
            ]

        except Exception as e:
            self.logger.error(f"Error in entity extraction: {str(e)}")
            return []

    async def analyze_conversation(self, 
                                 conversation: List[Dict]) -> Dict:
        """Analyze a conversation for patterns and metrics"""
        try:
            messages = len(conversation)
            sentiments = []
            intents = []

            for msg in conversation:
                if "content" in msg:
                    sentiment = await self.analyze_sentiment(msg["content"])
                    intent = await self.classify_intent(msg["content"])
                    sentiments.append(sentiment["label"])
                    intents.append(intent["intent"])

            return {
                "messages_count": messages,
                "sentiment_distribution": {
                    label: sentiments.count(label) / messages
                    for label in set(sentiments)
                },
                "intent_distribution": {
                    intent: intents.count(intent) / messages
                    for intent in set(intents)
                },
                "conversation_length": sum(len(msg.get("content", "")) 
                                        for msg in conversation)
            }

        except Exception as e:
            self.logger.error(f"Error analyzing conversation: {str(e)}")
            return {
                "error": str(e)
            }

    def get_model_info(self) -> Dict:
        """Get information about the NLP models"""
        return {
            "sentiment_analyzer": {
                "model": "distilbert-base-uncased-finetuned-sst-2-english",
                "loaded": self.sentiment_analyzer is not None
            },
            "intent_classifier": {
                "model": "facebook/bart-large-mnli",
                "loaded": self.intent_classifier is not None
            },
            "response_generator": {
                "model": "gpt2",
                "loaded": self.response_generator is not None
            },
            "version": "1.0.0"
        }
