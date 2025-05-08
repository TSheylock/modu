"""
Утилита командной строки для управления семантической сетью SASOK

Позволяет инициализировать, импортировать и запрашивать данные
из семантической сети SASOK через командную строку.
"""
import os
import sys
import asyncio
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv

from sasok_core.knowledge.semantic_network import SemanticNetwork
from sasok_core.knowledge.concept_net import ConceptNetProcessor
from sasok_core.knowledge.wordnet_processor import WordNetProcessor
from sasok_core.knowledge.wikidata_processor import WikidataProcessor
from sasok_core.knowledge.emotion_datasets import EmotionDatasetsProcessor

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SASOK.CLI")

# Получение настроек подключения к Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Функции для выполнения команд
async def initialize_network() -> SemanticNetwork:
    """
    Инициализация семантической сети
    
    Returns:
        Экземпляр SemanticNetwork
    """
    logger.info("Инициализация семантической сети...")
    
    # Создание и инициализация сети
    network = SemanticNetwork(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    await network.initialize()
    
    logger.info("Семантическая сеть инициализирована")
    return network

async def import_conceptnet(network: SemanticNetwork, args: argparse.Namespace):
    """
    Импорт данных из ConceptNet
    
    Args:
        network: Экземпляр семантической сети
        args: Аргументы командной строки
    """
    logger.info("Импорт данных из ConceptNet...")
    
    # Определение параметров импорта
    if args.emotions:
        # Импорт эмоциональных концептов
        emotions = args.emotions.split(",")
        max_depth = args.depth if args.depth else 2
        
        logger.info(f"Импорт эмоциональных концептов: {', '.join(emotions)} (глубина: {max_depth})")
        await network.concept_net.import_emotional_concepts(emotions, max_depth)
    elif args.concept:
        # Импорт концепта со связями
        max_depth = args.depth if args.depth else 2
        
        logger.info(f"Импорт концепта со связями: {args.concept} (глубина: {max_depth})")
        await network.concept_net.import_concept_with_relations(args.concept, max_depth)
    else:
        # Импорт базовых эмоциональных концептов
        emotions = ["радость", "грусть", "злость", "страх", "удивление", "отвращение"]
        max_depth = args.depth if args.depth else 1
        
        logger.info(f"Импорт базовых эмоциональных концептов (глубина: {max_depth})")
        await network.concept_net.import_emotional_concepts(emotions, max_depth)
    
    logger.info("Импорт данных из ConceptNet завершен")

async def import_wordnet(network: SemanticNetwork, args: argparse.Namespace):
    """
    Импорт данных из WordNet
    
    Args:
        network: Экземпляр семантической сети
        args: Аргументы командной строки
    """
    logger.info("Импорт данных из WordNet...")
    
    # Определение параметров импорта
    max_depth = args.depth if args.depth else 2
    
    # Импорт эмоциональной сети
    logger.info(f"Импорт эмоциональной сети WordNet (глубина: {max_depth})")
    await network.wordnet.import_emotion_network(max_depth)
    
    # Экспорт эмоционального лексикона
    if args.export_lexicon:
        output_file = args.output if args.output else None
        logger.info(f"Экспорт эмоционального лексикона WordNet")
        await network.wordnet.export_emotion_lexicon(output_file)
    
    logger.info("Импорт данных из WordNet завершен")

async def import_wikidata(network: SemanticNetwork, args: argparse.Namespace):
    """
    Импорт данных из Wikidata
    
    Args:
        network: Экземпляр семантической сети
        args: Аргументы командной строки
    """
    logger.info("Импорт данных из Wikidata...")
    
    # Определение параметров импорта
    max_depth = args.depth if args.depth else 1
    
    # Импорт эмоциональных сущностей
    logger.info("Импорт эмоциональных сущностей Wikidata")
    await network.wikidata.import_emotion_entities()
    
    # Импорт эмоциональной сети
    logger.info(f"Импорт эмоциональной сети Wikidata (глубина: {max_depth})")
    await network.wikidata.import_emotion_network(max_depth)
    
    logger.info("Импорт данных из Wikidata завершен")

async def import_emotion_datasets(network: SemanticNetwork, args: argparse.Namespace):
    """
    Импорт эмоциональных датасетов
    
    Args:
        network: Экземпляр семантической сети
        args: Аргументы командной строки
    """
    logger.info("Импорт эмоциональных датасетов...")
    
    # Импорт датасетов
    await network.emotion_datasets.import_all_datasets()
    
    # Обучение модели
    if args.train_model:
        model_name = args.model if args.model else "distilbert-base-uncased"
        dataset_name = args.dataset if args.dataset else "combined"
        output_dir = args.output if args.output else None
        
        logger.info(f"Обучение модели {model_name} на датасете {dataset_name}")
        await network.emotion_datasets.fine_tune_model(
            model_name=model_name,
            dataset_name=dataset_name,
            output_dir=output_dir
        )
    
    logger.info("Импорт эмоциональных датасетов завершен")

async def import_all(network: SemanticNetwork, args: argparse.Namespace):
    """
    Импорт данных из всех источников
    
    Args:
        network: Экземпляр семантической сети
        args: Аргументы командной строки
    """
    logger.info("Импорт данных из всех источников...")
    
    # Определение параметров импорта
    max_depth = args.depth if args.depth else 1
    
    # Запуск импорта в параллельных задачах
    tasks = [
        network.concept_net.import_emotional_concepts(
            ["радость", "грусть", "злость", "страх", "удивление", "отвращение"],
            max_depth
        ),
        network.wordnet.import_emotion_network(max_depth),
        network.wikidata.import_emotion_entities(),
        network.emotion_datasets.import_all_datasets()
    ]
    
    await asyncio.gather(*tasks)
    
    logger.info("Импорт данных из всех источников завершен")

async def analyze_concept(network: SemanticNetwork, args: argparse.Namespace):
    """
    Анализ эмоций концепта
    
    Args:
        network: Экземпляр семантической сети
        args: Аргументы командной строки
    """
    if not args.concept:
        logger.error("Не указан концепт для анализа")
        return
    
    # Определение типа эмоций
    emotion_type = args.emotion_type if args.emotion_type else "basic"
    
    logger.info(f"Анализ эмоций концепта '{args.concept}' (тип: {emotion_type})...")
    
    # Получение эмоционального профиля
    emotion_profile = await network.analyze_concept_emotions(args.concept, emotion_type)
    
    # Вывод результатов
    print(f"\nЭмоциональный профиль концепта '{args.concept}':")
    for emotion, weight in sorted(emotion_profile.items(), key=lambda x: x[1], reverse=True):
        if weight > 0:
            print(f"  {emotion}: {weight:.3f}")
    
    # Сохранение результатов в файл
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "concept": args.concept,
                "emotion_type": emotion_type,
                "emotion_profile": emotion_profile
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Результаты сохранены в файл: {args.output}")

async def analyze_text(network: SemanticNetwork, args: argparse.Namespace):
    """
    Анализ эмоций текста
    
    Args:
        network: Экземпляр семантической сети
        args: Аргументы командной строки
    """
    text = ""
    
    if args.text:
        text = args.text
    elif args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Ошибка чтения файла: {e}")
            return
    else:
        logger.error("Не указан текст или файл для анализа")
        return
    
    logger.info(f"Анализ эмоций текста (длина: {len(text)} символов)...")
    
    # Получение эмоционального профиля
    emotion_profile = await network.build_emotional_profile(text)
    
    # Вывод результатов
    print(f"\nЭмоциональный профиль текста:")
    for emotion, weight in sorted(emotion_profile.items(), key=lambda x: x[1], reverse=True):
        if weight > 0:
            print(f"  {emotion}: {weight:.3f}")
    
    # Сохранение результатов в файл
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "emotion_profile": emotion_profile
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Результаты сохранены в файл: {args.output}")

async def find_path(network: SemanticNetwork, args: argparse.Namespace):
    """
    Поиск пути между концептами
    
    Args:
        network: Экземпляр семантической сети
        args: Аргументы командной строки
    """
    if not args.source or not args.target:
        logger.error("Не указаны исходный или целевой концепты")
        return
    
    # Определение параметров поиска
    max_depth = args.depth if args.depth else 3
    
    logger.info(f"Поиск пути от '{args.source}' к '{args.target}' (глубина: {max_depth})...")
    
    # Поиск путей в разных источниках
    paths = {}
    
    if not args.source or args.source == "all":
        # Поиск в ConceptNet
        concept_path = await network.concept_net.find_emotional_path(args.source, args.target, max_depth)
        paths["conceptnet"] = concept_path
        
        # Поиск в WordNet
        wordnet_path = await network.wordnet.find_emotion_path(args.source, args.target, max_depth)
        paths["wordnet"] = wordnet_path
        
        # Поиск в Wikidata
        wikidata_path = await network.wikidata.find_emotion_path(args.source, args.target, max_depth)
        paths["wikidata"] = wikidata_path
    else:
        # Поиск в указанном источнике
        if args.source == "conceptnet":
            path = await network.concept_net.find_emotional_path(args.source, args.target, max_depth)
            paths["conceptnet"] = path
        elif args.source == "wordnet":
            path = await network.wordnet.find_emotion_path(args.source, args.target, max_depth)
            paths["wordnet"] = path
        elif args.source == "wikidata":
            path = await network.wikidata.find_emotion_path(args.source, args.target, max_depth)
            paths["wikidata"] = path
    
    # Вывод результатов
    print(f"\nПути от '{args.source}' к '{args.target}':")
    for source, path in paths.items():
        if path:
            print(f"\n[{source.upper()}]")
            for i, item in enumerate(path):
                print(f"  {i+1}. {item}")
        else:
            print(f"\n[{source.upper()}] Путь не найден")
    
    # Сохранение результатов в файл
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "source": args.source,
                "target": args.target,
                "max_depth": max_depth,
                "paths": paths
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Результаты сохранены в файл: {args.output}")

# Основная функция
async def main():
    # Создание парсера аргументов
    parser = argparse.ArgumentParser(description="Управление семантической сетью SASOK")
    subparsers = parser.add_subparsers(dest="command", help="Команда")
    
    # Команда импорта ConceptNet
    conceptnet_parser = subparsers.add_parser("import-conceptnet", help="Импорт данных из ConceptNet")
    conceptnet_parser.add_argument("--concept", help="Концепт для импорта")
    conceptnet_parser.add_argument("--emotions", help="Список эмоций для импорта (через запятую)")
    conceptnet_parser.add_argument("--depth", type=int, help="Глубина импорта")
    
    # Команда импорта WordNet
    wordnet_parser = subparsers.add_parser("import-wordnet", help="Импорт данных из WordNet")
    wordnet_parser.add_argument("--depth", type=int, help="Глубина импорта")
    wordnet_parser.add_argument("--export-lexicon", action="store_true", help="Экспорт эмоционального лексикона")
    wordnet_parser.add_argument("--output", help="Путь к выходному файлу")
    
    # Команда импорта Wikidata
    wikidata_parser = subparsers.add_parser("import-wikidata", help="Импорт данных из Wikidata")
    wikidata_parser.add_argument("--depth", type=int, help="Глубина импорта")
    
    # Команда импорта эмоциональных датасетов
    datasets_parser = subparsers.add_parser("import-datasets", help="Импорт эмоциональных датасетов")
    datasets_parser.add_argument("--train-model", action="store_true", help="Обучение модели на датасетах")
    datasets_parser.add_argument("--model", help="Название модели для обучения")
    datasets_parser.add_argument("--dataset", help="Название датасета для обучения")
    datasets_parser.add_argument("--output", help="Путь к директории для сохранения модели")
    
    # Команда импорта всех источников
    all_parser = subparsers.add_parser("import-all", help="Импорт данных из всех источников")
    all_parser.add_argument("--depth", type=int, help="Глубина импорта")
    
    # Команда анализа эмоций концепта
    concept_parser = subparsers.add_parser("analyze-concept", help="Анализ эмоций концепта")
    concept_parser.add_argument("concept", help="Концепт для анализа")
    concept_parser.add_argument("--emotion-type", choices=["basic", "complex", "all"], help="Тип эмоций")
    concept_parser.add_argument("--output", help="Путь к выходному файлу")
    
    # Команда анализа эмоций текста
    text_parser = subparsers.add_parser("analyze-text", help="Анализ эмоций текста")
    text_parser.add_argument("--text", help="Текст для анализа")
    text_parser.add_argument("--file", help="Путь к файлу с текстом")
    text_parser.add_argument("--output", help="Путь к выходному файлу")
    
    # Команда поиска пути между концептами
    path_parser = subparsers.add_parser("find-path", help="Поиск пути между концептами")
    path_parser.add_argument("source", help="Исходный концепт")
    path_parser.add_argument("target", help="Целевой концепт")
    path_parser.add_argument("--source-type", choices=["conceptnet", "wordnet", "wikidata", "all"], default="all", help="Источник для поиска")
    path_parser.add_argument("--depth", type=int, help="Глубина поиска")
    path_parser.add_argument("--output", help="Путь к выходному файлу")
    
    # Парсинг аргументов
    args = parser.parse_args()
    
    # Инициализация семантической сети
    network = await initialize_network()
    
    try:
        # Выполнение команды
        if args.command == "import-conceptnet":
            await import_conceptnet(network, args)
        elif args.command == "import-wordnet":
            await import_wordnet(network, args)
        elif args.command == "import-wikidata":
            await import_wikidata(network, args)
        elif args.command == "import-datasets":
            await import_emotion_datasets(network, args)
        elif args.command == "import-all":
            await import_all(network, args)
        elif args.command == "analyze-concept":
            await analyze_concept(network, args)
        elif args.command == "analyze-text":
            await analyze_text(network, args)
        elif args.command == "find-path":
            await find_path(network, args)
        else:
            parser.print_help()
    except Exception as e:
        logger.error(f"Ошибка выполнения команды: {e}")
    finally:
        # Закрытие соединений
        network.close()

if __name__ == "__main__":
    asyncio.run(main())
