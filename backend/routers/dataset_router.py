from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional
from ..ml.dataset_loaders import DatasetLoaders
from ..ml.dataset_manager import DatasetManager
from ..services.web3_service import Web3Service
from ..config import config
import asyncio
import logging
from datetime import datetime

router = APIRouter(prefix="/datasets", tags=["datasets"])
logger = logging.getLogger(__name__)

# Initialize loaders and manager
dataset_loaders = DatasetLoaders()
dataset_manager = DatasetManager(config)

class DatasetRequest(BaseModel):
    name: str
    save_to_ipfs: bool = True
    user_address: Optional[str] = None
    signature: Optional[str] = None

class DatasetLoadStatus(BaseModel):
    dataset_name: str
    status: str
    progress: float
    error: Optional[str] = None

# Store loading status
loading_status = {}

def get_web3_service():
    return Web3Service(config)

@router.get("/available")
async def get_available_datasets():
    """Get list of available datasets"""
    try:
        datasets = dataset_loaders.get_available_datasets()
        return {
            "success": True,
            "datasets": datasets
        }
    except Exception as e:
        logger.error(f"Error getting available datasets: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/load")
async def load_dataset(
    request: DatasetRequest,
    background_tasks: BackgroundTasks,
    web3_service: Web3Service = Depends(get_web3_service)
):
    """Load a specific dataset"""
    try:
        # Initialize loading status
        loading_status[request.name] = DatasetLoadStatus(
            dataset_name=request.name,
            status="starting",
            progress=0.0
        )

        # Start loading in background
        background_tasks.add_task(
            load_dataset_task,
            request,
            web3_service
        )

        return {
            "success": True,
            "message": f"Started loading dataset {request.name}",
            "status_endpoint": f"/datasets/status/{request.name}"
        }

    except Exception as e:
        logger.error(f"Error initiating dataset load: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def load_dataset_task(request: DatasetRequest, web3_service: Web3Service):
    """Background task for loading dataset"""
    try:
        # Update status
        loading_status[request.name].status = "loading"
        loading_status[request.name].progress = 0.1

        # Load dataset
        if request.name == "GoEmotions":
            result = await dataset_loaders.load_go_emotions()
        elif request.name == "IEMOCAP":
            result = await dataset_loaders.load_iemocap()
        elif request.name == "EmoBank":
            result = await dataset_loaders.load_emobank()
        # Add other dataset conditions here
        else:
            raise ValueError(f"Unknown dataset: {request.name}")

        if not result["success"]:
            raise Exception(result.get("error", "Unknown error"))

        loading_status[request.name].progress = 0.5

        # Save to IPFS if requested
        if request.save_to_ipfs:
            ipfs_result = await dataset_manager.save_to_ipfs(
                result["data"],
                request.name
            )
            if not ipfs_result["success"]:
                raise Exception(ipfs_result.get("error", "IPFS save failed"))

            loading_status[request.name].progress = 0.8

            # Record on blockchain if user provided
            if request.user_address and request.signature:
                tx_result = await web3_service.record_interaction(
                    request.user_address,
                    "DATASET_LOAD",
                    {
                        "dataset": request.name,
                        "ipfs_hash": ipfs_result["ipfs_hash"],
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                if not tx_result["success"]:
                    raise Exception(tx_result.get("error", "Blockchain record failed"))

        loading_status[request.name].status = "completed"
        loading_status[request.name].progress = 1.0

    except Exception as e:
        logger.error(f"Error loading dataset {request.name}: {str(e)}")
        loading_status[request.name].status = "failed"
        loading_status[request.name].error = str(e)

@router.get("/status/{dataset_name}")
async def get_loading_status(dataset_name: str):
    """Get dataset loading status"""
    try:
        if dataset_name not in loading_status:
            raise HTTPException(
                status_code=404,
                detail=f"No loading status for dataset {dataset_name}"
            )

        return loading_status[dataset_name]

    except Exception as e:
        logger.error(f"Error getting loading status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/info/{dataset_name}")
async def get_dataset_info(dataset_name: str):
    """Get information about a loaded dataset"""
    try:
        info = await dataset_manager.get_dataset_info(dataset_name)
        if not info["success"]:
            raise HTTPException(
                status_code=404,
                detail=info.get("error", f"Dataset {dataset_name} not found")
            )

        return info

    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/prepare")
async def prepare_training_data(
    dataset_name: str,
    split: str = "train"
):
    """Prepare dataset for training"""
    try:
        result = await dataset_manager.prepare_training_data(dataset_name, split)
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Failed to prepare training data")
            )

        return result

    except Exception as e:
        logger.error(f"Error preparing training data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{dataset_name}")
async def remove_dataset(
    dataset_name: str,
    user_address: Optional[str] = None,
    signature: Optional[str] = None,
    web3_service: Web3Service = Depends(get_web3_service)
):
    """Remove a dataset"""
    try:
        # Verify user if provided
        if user_address and signature:
            profile = await web3_service.get_user_profile(user_address)
            if not profile["success"] or not profile["profile"]["isRegistered"]:
                raise HTTPException(
                    status_code=401,
                    detail="User not registered on blockchain"
                )

        # Remove dataset
        result = await dataset_manager.remove_dataset(dataset_name)
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=result.get("error", f"Failed to remove dataset {dataset_name}")
            )

        # Record on blockchain if user provided
        if user_address and signature:
            await web3_service.record_interaction(
                user_address,
                "DATASET_REMOVE",
                {
                    "dataset": dataset_name,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

        return {
            "success": True,
            "message": f"Dataset {dataset_name} removed successfully"
        }

    except Exception as e:
        logger.error(f"Error removing dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
