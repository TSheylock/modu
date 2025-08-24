"""
API для доступа к графу знаний SASOK

Предоставляет эндпоинты для получения данных о семантической сети
для визуализации в UI с использованием Cytoscape.js.
"""
import os
import logging
import json
from typing import Dict, List, Any

from fastapi import APIRouter, Path, Query, HTTPException
from fastapi.responses import JSONResponse
from py2neo import Graph

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SASOK.GraphAPI")

# Настройки Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Инициализация роутера
router = APIRouter(prefix="/api/graph", tags=["Graph"])

# Подключение к Neo4j
try:
    graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    logger.info(f"Подключение к Neo4j успешно: {NEO4J_URI}")
except Exception as e:
    logger.error(f"Ошибка подключения к Neo4j: {e}")
    graph = None

@router.get("/")
async def get_graph_status():
    """
    Проверка состояния графа знаний
    """
    if not graph:
        return {"status": "error", "message": "Нет подключения к Neo4j"}
    
    try:
        # Подсчет количества узлов и связей
        query = """
        MATCH (n) 
        RETURN count(n) AS nodes
        """
        nodes_result = graph.run(query).data()
        node_count = nodes_result[0]["nodes"] if nodes_result else 0
        
        query = """
        MATCH ()-[r]->() 
        RETURN count(r) AS relationships
        """
        rel_result = graph.run(query).data()
        rel_count = rel_result[0]["relationships"] if rel_result else 0
        
        return {
            "status": "ok",
            "nodes": node_count,
            "relationships": rel_count,
            "message": "Граф знаний SASOK активен"
        }
    except Exception as e:
        logger.error(f"Ошибка проверки состояния графа: {e}")
        return {"status": "error", "message": str(e)}

@router.get("/data")
async def get_graph_data(
    depth: int = Query(1, description="Глубина обхода графа"),
    start_node: str = Query(None, description="Начальный узел (имя концепта или эмоции)"),
    node_type: str = Query(None, description="Тип узла (Concept, Emotion, Property)")
):
    """
    Получение данных графа для визуализации
    """
    if not graph:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Нет подключения к Neo4j"}
        )
    
    try:
        # Формирование запроса в зависимости от параметров
        if start_node and node_type:
            # Поиск от конкретного узла
            query = f"""
            MATCH path = (start:{node_type} {{name: $start_node}})-[*1..{depth}]-(related)
            RETURN path
            LIMIT 100
            """
            parameters = {"start_node": start_node}
        elif node_type:
            # Поиск всех узлов определенного типа с связями
            query = f"""
            MATCH path = (n:{node_type})-[*1..{depth}]-(related)
            RETURN path
            LIMIT 100
            """
            parameters = {}
        else:
            # Общий обзор графа
            query = f"""
            MATCH path = (n)-[*1..{depth}]-(related)
            RETURN path
            LIMIT 100
            """
            parameters = {}
        
        # Выполнение запроса
        result = graph.run(query, **parameters).data()
        
        # Преобразование результата в формат для Cytoscape.js
        cyto_nodes = {}
        cyto_edges = {}
        
        for record in result:
            path = record["path"]
            
            # Обработка узлов
            for node in path.nodes:
                if node.identity not in cyto_nodes:
                    # Определение цвета узла по метке
                    node_color = "#6FB1FC"  # По умолчанию
                    
                    if "Emotion" in node.labels:
                        if "basic" in node.get("type", ""):
                            node_color = "#FF5733"  # Базовые эмоции
                        else:
                            node_color = "#FFC300"  # Сложные эмоции
                    elif "Concept" in node.labels:
                        node_color = "#33FF57"  # Концепты
                    elif "Property" in node.labels:
                        node_color = "#33A8FF"  # Свойства
                    
                    # Добавление узла
                    cyto_nodes[node.identity] = {
                        "data": {
                            "id": str(node.identity),
                            "name": node.get("name", ""),
                            "label": next(iter(node.labels), ""),
                            "type": node.get("type", ""),
                            "color": node_color,
                            "properties": dict(node)
                        }
                    }
            
            # Обработка связей
            for rel in path.relationships:
                edge_id = f"{rel.start_node.identity}-{rel.type}-{rel.end_node.identity}"
                
                if edge_id not in cyto_edges:
                    # Добавление связи
                    cyto_edges[edge_id] = {
                        "data": {
                            "id": edge_id,
                            "source": str(rel.start_node.identity),
                            "target": str(rel.end_node.identity),
                            "label": rel.type,
                            "weight": rel.get("weight", 1.0),
                            "properties": dict(rel)
                        }
                    }
        
        # Формирование итогового графа
        cytoscape_data = {
            "nodes": list(cyto_nodes.values()),
            "edges": list(cyto_edges.values())
        }
        
        return cytoscape_data
    
    except Exception as e:
        logger.error(f"Ошибка получения данных графа: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@router.get("/concepts")
async def get_concepts():
    """
    Получение списка всех концептов
    """
    if not graph:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Нет подключения к Neo4j"}
        )
    
    try:
        query = """
        MATCH (c:Concept)
        RETURN c.name AS name
        ORDER BY c.name
        """
        
        result = graph.run(query).data()
        concepts = [item["name"] for item in result]
        
        return {"concepts": concepts}
    
    except Exception as e:
        logger.error(f"Ошибка получения концептов: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@router.get("/emotions")
async def get_emotions():
    """
    Получение списка всех эмоций
    """
    if not graph:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Нет подключения к Neo4j"}
        )
    
    try:
        query = """
        MATCH (e:Emotion)
        RETURN e.name AS name, e.type AS type
        ORDER BY e.type, e.name
        """
        
        result = graph.run(query).data()
        
        # Группировка эмоций по типу
        emotions = {
            "basic": [],
            "complex": [],
            "derived": []
        }
        
        for item in result:
            emotion_type = item.get("type", "derived")
            if emotion_type not in emotions:
                emotions[emotion_type] = []
            
            emotions[emotion_type].append(item["name"])
        
        return emotions
    
    except Exception as e:
        logger.error(f"Ошибка получения эмоций: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@router.get("/emotion-profile/{concept}")
async def get_emotion_profile(concept: str = Path(..., description="Имя концепта")):
    """
    Получение эмоционального профиля концепта
    """
    if not graph:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Нет подключения к Neo4j"}
        )
    
    try:
        # Запрос к Neo4j для получения эмоционального профиля
        query = """
        MATCH (c:Concept {name: $concept})-[r:EVOKES]->(e:Emotion)
        RETURN e.name AS emotion, r.weight AS weight, e.type AS type
        UNION
        MATCH (c:Concept {name: $concept})-[:HAS_PROPERTY]->(p:Property),
                (e:Emotion)-[:RELATED_TO]-(p)
        RETURN e.name AS emotion, 0.5 AS weight, e.type AS type
        """
        
        result = graph.run(query, concept=concept).data()
        
        # Формирование эмоционального профиля
        emotion_profile = {}
        
        for item in result:
            emotion = item["emotion"]
            weight = item["weight"]
            emotion_type = item.get("type", "derived")
            
            if emotion in emotion_profile:
                # Если эмоция уже есть, берем максимальный вес
                emotion_profile[emotion] = {
                    "weight": max(emotion_profile[emotion]["weight"], weight),
                    "type": emotion_type
                }
            else:
                emotion_profile[emotion] = {
                    "weight": weight,
                    "type": emotion_type
                }
        
        # Сортировка по весу (от большего к меньшему)
        sorted_profile = {}
        for emotion, data in sorted(
            emotion_profile.items(), 
            key=lambda x: x[1]["weight"], 
            reverse=True
        ):
            sorted_profile[emotion] = data
        
        return {
            "concept": concept,
            "emotions": sorted_profile
        }
    
    except Exception as e:
        logger.error(f"Ошибка получения эмоционального профиля: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@router.get("/similar-concepts/{concept}")
async def get_similar_concepts(
    concept: str = Path(..., description="Имя концепта"),
    limit: int = Query(5, description="Количество похожих концептов")
):
    """
    Получение похожих концептов на основе общих эмоций и свойств
    """
    if not graph:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Нет подключения к Neo4j"}
        )
    
    try:
        # Запрос к Neo4j для получения похожих концептов
        query = """
        MATCH (c1:Concept {name: $concept})-[:EVOKES]->(e:Emotion)<-[:EVOKES]-(c2:Concept)
        WHERE c1 <> c2
        WITH c2, count(e) AS common_emotions
        RETURN c2.name AS concept, common_emotions
        ORDER BY common_emotions DESC
        LIMIT $limit
        """
        
        result = graph.run(query, concept=concept, limit=limit).data()
        
        # Формирование списка похожих концептов
        similar_concepts = []
        
        for item in result:
            similar_concepts.append({
                "name": item["concept"],
                "common_emotions": item["common_emotions"],
                "similarity_score": item["common_emotions"] / 10.0  # Нормализация оценки
            })
        
        return {
            "concept": concept,
            "similar_concepts": similar_concepts
        }
    
    except Exception as e:
        logger.error(f"Ошибка получения похожих концептов: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@router.get("/path/{source_concept}/{target_emotion}")
async def find_emotion_path(
    source_concept: str = Path(..., description="Исходный концепт"),
    target_emotion: str = Path(..., description="Целевая эмоция"),
    max_depth: int = Query(3, description="Максимальная глубина поиска")
):
    """
    Поиск пути от концепта к эмоции
    """
    if not graph:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Нет подключения к Neo4j"}
        )
    
    try:
        # Запрос к Neo4j для поиска пути
        query = f"""
        MATCH path = shortestPath((c:Concept {{name: $source}})-[*1..{max_depth}]-(e:Emotion {{name: $target}}))
        RETURN path
        LIMIT 1
        """
        
        result = graph.run(query, source=source_concept, target=target_emotion).data()
        
        if not result:
            return {
                "source": source_concept,
                "target": target_emotion,
                "path_found": False,
                "path": []
            }
        
        # Извлечение пути
        path = result[0]["path"]
        
        # Преобразование пути в список узлов и отношений
        path_nodes = []
        path_edges = []
        
        # Обработка узлов
        for node in path.nodes:
            path_nodes.append({
                "id": str(node.identity),
                "name": node.get("name", ""),
                "label": next(iter(node.labels), ""),
                "type": node.get("type", ""),
                "properties": dict(node)
            })
        
        # Обработка связей
        for rel in path.relationships:
            path_edges.append({
                "id": f"{rel.start_node.identity}-{rel.type}-{rel.end_node.identity}",
                "source": str(rel.start_node.identity),
                "target": str(rel.end_node.identity),
                "label": rel.type,
                "weight": rel.get("weight", 1.0),
                "properties": dict(rel)
            })
        
        return {
            "source": source_concept,
            "target": target_emotion,
            "path_found": True,
            "path_length": len(path_nodes),
            "nodes": path_nodes,
            "edges": path_edges
        }
    
    except Exception as e:
        logger.error(f"Ошибка поиска пути: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
