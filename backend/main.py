from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from datetime import datetime
import json
import os
from neo4j import GraphDatabase
import spacy

# Load spaCy model (ensure 'en_core_web_sm' is downloaded)
nlp = spacy.load('en_core_web_sm')

# Create FastAPI app
app = FastAPI(title="SASOK Backend API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Neo4j driver setup for graph data
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

API_KEY = os.getenv('API_KEY', 'default_key')  # Set via environment variable

async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# Models
class EmotionAnalysis(BaseModel):
    timestamp: datetime
    emotion: str
    confidence: float
    user_id: str

class Web3Transaction(BaseModel):
    transaction_hash: str
    from_address: str
    to_address: str
    value: str
    timestamp: datetime

class UserInteraction(BaseModel):
    user_id: str
    interaction_type: str
    data: dict
    timestamp: datetime

# Function to add or update a node
async def add_or_update_node(node_data: dict):
    with driver.session() as session:
        session.run("MERGE (n:Node {id: $id}) SET n.label = $label, n.type = $type, n.valence = $valence, n.intensity = $intensity RETURN n", node_data)
    return {"status": "node added or updated"}

# Function to add or update an edge
async def add_or_update_edge(edge_data: dict):
    with driver.session() as session:
        session.run("MATCH (a:Node {id: $source}), (b:Node {id: $target}) MERGE (a)-[r:RELATIONSHIP {type: $type}]->(b) SET r.weight = $weight, r.timestamp = $timestamp RETURN r", edge_data)
    return {"status": "edge added or updated"}

# Routes for AI Processing
@app.post("/api/ai/emotion-analysis")
async def analyze_emotion(data: dict):
    try:
        # Placeholder for emotion analysis logic
        return {
            "status": "success",
            "emotion": "neutral",
            "confidence": 0.85
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/nlp-process")
async def process_nlp(text: str):
    try:
        # Placeholder for NLP processing logic
        return {
            "status": "success",
            "intent": "query",
            "entities": [],
            "sentiment": "neutral"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# NLP-based endpoint to process message and update graph
@app.post("/api/nlp/process-and-update", dependencies=[Depends(verify_api_key)])
async def process_and_update(message: str, x_api_key: str = Depends(verify_api_key)):
    doc = nlp(message)
    entities = [ent.text for ent in doc.ents]  # Extract entities
    # Simplified logic: create/update nodes for each entity and add edges if needed
    for entity in entities:
        node_data = {"id": entity, "label": entity, "type": "concept", "valence": 0.0, "intensity": 1.0}  # Default values; refine based on context
        await add_or_update_node(node_data)  # Add or update node
        # Example edge creation: link to a default 'user' node or based on message context
        edge_data = {"source": "user", "target": entity, "type": "associated", "weight": 1.0, "timestamp": datetime.utcnow().isoformat()}
        await add_or_update_edge(edge_data)
    return {"status": "graph updated", "entities_found": entities}

# Routes for Web3 Integration
@app.post("/api/web3/connect")
async def connect_wallet(wallet_address: str):
    try:
        # Placeholder for wallet connection logic
        return {
            "status": "connected",
            "address": wallet_address,
            "network": "ethereum"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/web3/transactions/{address}")
async def get_transactions(address: str):
    try:
        # Placeholder for transaction fetching logic
        return {
            "status": "success",
            "transactions": []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Routes for Analytics
@app.post("/api/analytics/log")
async def log_interaction(interaction: UserInteraction):
    try:
        # Placeholder for interaction logging logic
        return {
            "status": "success",
            "interaction_id": "generated_id"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/stats")
async def get_stats():
    try:
        # Placeholder for analytics stats
        return {
            "active_users": 1234,
            "total_interactions": 5678,
            "emotion_distribution": {
                "happy": 45,
                "neutral": 30,
                "sad": 25
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# New route for graph data
@app.get("/api/graph/data")
async def get_graph_data():
    with driver.session() as session:
        nodes_result = session.run("MATCH (n) RETURN id(n) as id, labels(n) as labels, properties(n) as properties")
        edges_result = session.run("MATCH ()-[r]-() RETURN id(r) as id, type(r) as type, startNode(r) as start, endNode(r) as end, properties(r) as properties")
        nodes = [record.data() for record in nodes_result]
        edges = [record.data() for record in edges_result]
        return {"nodes": nodes, "edges": edges}

# Endpoint for updating graph based on user feedback
@app.post("/api/graph/update-feedback", dependencies=[Depends(verify_api_key)])
async def update_feedback(feedback_data: dict, x_api_key: str = Depends(verify_api_key)):
    try:
        node_id = feedback_data.get('node_id')
        edge_type = feedback_data.get('edge_type')
        feedback = feedback_data.get('feedback')
        weight_change = feedback_data.get('weight_change', 0.1) if feedback == 'positive' else -feedback_data.get('weight_change', 0.1)
        
        # Update node intensity or other attributes
        if node_id:
            with driver.session() as session:
                session.run("MATCH (n:Node {id: $node_id}) SET n.intensity = n.intensity + $weight_change RETURN n", {'node_id': node_id, 'weight_change': weight_change})
        
        # Update edge weight
        if edge_type:
            with driver.session() as session:
                session.run("MATCH ()-[r:RELATIONSHIP {type: $edge_type}]->() SET r.weight = r.weight + $weight_change RETURN r", {'edge_type': edge_type, 'weight_change': weight_change})
        
        return {"status": "graph updated based on feedback"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for updating a node
@app.post("/api/graph/update-node", dependencies=[Depends(verify_api_key)])
async def update_node(node_data: dict, x_api_key: str = Depends(verify_api_key)):
    try:
        with driver.session() as session:
            session.run("MATCH (n:Node {id: $node_id}) SET n += $properties RETURN n", {'node_id': node_data['node_id'], 'properties': node_data})
        return {"status": "node updated"}
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing key: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
