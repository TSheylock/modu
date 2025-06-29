from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Optional
from datetime import datetime
from ..quantum.memory_core import QuantumMemoryCore
from ..services.web3_service import Web3Service
from ..config import config

router = APIRouter(prefix="/quantum", tags=["quantum"])
memory_core = QuantumMemoryCore()

class PerceptionRequest(BaseModel):
    user_input: str
    user_address: str
    signature: str

class RetrieveRequest(BaseModel):
    node_name: str
    user_address: str
    signature: str

class ConnectionRequest(BaseModel):
    node1: str
    node2: str
    weight: Optional[float] = 1.0
    user_address: str
    signature: str

def get_web3_service():
    return Web3Service(config)

@router.post("/perception")
async def process_perception(
    request: PerceptionRequest,
    web3_service: Web3Service = Depends(get_web3_service)
):
    """Process and store quantum perception data"""
    try:
        # Verify user is registered on blockchain
        user_profile = await web3_service.get_user_profile(request.user_address)
        if not user_profile['success'] or not user_profile['profile']['isRegistered']:
            raise HTTPException(
                status_code=401,
                detail="User not registered on blockchain"
            )

        # Process perception through quantum core
        perception_data = memory_core.quantum_perception(request.user_input)
        
        # Store perception with blockchain verification
        result = await memory_core.store_perception(
            request.user_address,
            perception_data,
            request.signature
        )

        # Record interaction on blockchain
        interaction_data = {
            "type": "quantum_perception",
            "metadata": {
                "node_name": result["node_name"],
                "timestamp": result["timestamp"]
            }
        }
        
        tx_result = await web3_service.record_interaction(
            request.user_address,
            "QUANTUM_PERCEPTION",
            str(interaction_data)
        )

        return {
            "success": True,
            "perception_result": result,
            "blockchain_tx": tx_result['transaction']['hash'] if tx_result['success'] else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retrieve")
async def retrieve_perception(
    request: RetrieveRequest,
    web3_service: Web3Service = Depends(get_web3_service)
):
    """Retrieve stored quantum perception data"""
    try:
        # Verify user is registered
        user_profile = await web3_service.get_user_profile(request.user_address)
        if not user_profile['success'] or not user_profile['profile']['isRegistered']:
            raise HTTPException(
                status_code=401,
                detail="User not registered on blockchain"
            )

        # Retrieve perception data
        result = await memory_core.retrieve_perception(
            request.node_name,
            request.user_address,
            request.signature
        )

        # Record retrieval interaction
        interaction_data = {
            "type": "retrieve_perception",
            "metadata": {
                "node_name": request.node_name,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        await web3_service.record_interaction(
            request.user_address,
            "RETRIEVE_PERCEPTION",
            str(interaction_data)
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/connect")
async def connect_nodes(
    request: ConnectionRequest,
    web3_service: Web3Service = Depends(get_web3_service)
):
    """Create connection between memory nodes"""
    try:
        # Verify user is registered
        user_profile = await web3_service.get_user_profile(request.user_address)
        if not user_profile['success'] or not user_profile['profile']['isRegistered']:
            raise HTTPException(
                status_code=401,
                detail="User not registered on blockchain"
            )

        # Add connection in memory core
        memory_core.add_connection(
            request.node1,
            request.node2,
            request.weight
        )

        # Record connection on blockchain
        interaction_data = {
            "type": "connect_nodes",
            "metadata": {
                "node1": request.node1,
                "node2": request.node2,
                "weight": request.weight,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        tx_result = await web3_service.record_interaction(
            request.user_address,
            "CONNECT_NODES",
            str(interaction_data)
        )

        return {
            "success": True,
            "connection": {
                "node1": request.node1,
                "node2": request.node2,
                "weight": request.weight
            },
            "blockchain_tx": tx_result['transaction']['hash'] if tx_result['success'] else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/memory")
async def get_memory_state():
    """Get current state of quantum memory core"""
    try:
        return memory_core.export_to_json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/node/{node_name}")
async def get_node_info(node_name: str):
    """Get detailed information about a specific node"""
    try:
        return memory_core.get_node_connections(node_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def check_health():
    """Check health of quantum memory system"""
    try:
        memory_state = memory_core.export_to_json()
        return {
            "status": "healthy",
            "nodes_count": len(memory_state["nodes"]),
            "edges_count": len(memory_state["edges"]),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Memory core health check failed: {str(e)}"
        )
