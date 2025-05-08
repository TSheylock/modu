"""
Пакет интеграции баз знаний и семантической сети SASOK

Обеспечивает загрузку, обработку и интеграцию данных из различных
источников знаний (ConceptNet, WordNet, Wikidata) и эмоциональных датасетов
в единую семантическую сеть для эмоционального анализа.
"""

from sasok_core.knowledge.knowledge_base import KnowledgeBase
from sasok_core.knowledge.concept_net import ConceptNetProcessor
from sasok_core.knowledge.wordnet_processor import WordNetProcessor
from sasok_core.knowledge.wikidata_processor import WikidataProcessor
from sasok_core.knowledge.emotion_datasets import EmotionDatasetsProcessor
from sasok_core.knowledge.semantic_network import SemanticNetwork

__all__ = [
    'KnowledgeBase',
    'ConceptNetProcessor',
    'WordNetProcessor',
    'WikidataProcessor',
    'EmotionDatasetsProcessor',
    'SemanticNetwork'
]
