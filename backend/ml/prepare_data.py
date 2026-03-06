
import asyncio
import numpy as np
from dataset_manager import DatasetManager
from data_pipeline import MultiModalPipeline

async def main():
    # 1. Load GoEmotions dataset
    dataset_manager = DatasetManager(config={})
    go_emotions_data = await dataset_manager.load_dataset("GoEmotions")

    if not go_emotions_data.get("success"):
        print("Failed to load GoEmotions dataset")
        return

    print("GoEmotions dataset loaded successfully.")
    
    # 2. Process text and create npz
    pipeline = MultiModalPipeline()
    
    texts = go_emotions_data["data"]["train"]["texts"]
    emotions = go_emotions_data["data"]["train"]["emotions"]
    
    embeddings = []
    labels = []
    
    print("Processing texts...")
    for i, text in enumerate(texts):
        processed_text = await pipeline.process_text(text)
        if processed_text.get("success"):
            embeddings.append(processed_text["embeddings"])
            # Use the first emotion as the label
            label = np.argmax(emotions[i])
            labels.append(label)

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    print(f"Processed {len(embeddings)} texts.")
    
    # 3. Save to .npz file
    np.savez_compressed('data/goemotions.npz', X=embeddings, y=labels)
    print("Saved data to data/goemotions.npz")


if __name__ == "__main__":
    asyncio.run(main())
