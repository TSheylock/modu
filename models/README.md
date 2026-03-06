# SASOK Trained Models

## Required Models (not in git — too large)

| File | Size | Description |
|------|------|-------------|
| `audio_model.pt` | 1.26 GB | Trained audio emotion recognition model |
| `emotion_model.torchscript` | 328 MB | TorchScript emotion classification model |

## How to Obtain

Copy from main working directory:

```bash
cp ~/modu/models/audio_model.pt ./models/
cp ~/modu/models/emotion_model.torchscript ./models/
```

Or retrain from scratch — see `backend/ml/train_emotion_cnn.py`.

## Training Metadata

- Epochs: 5
- Accuracy: 86%
- Final Loss: 0.28
- Data Sources: webcam (30), text (22), audio (22)
- Dataset: FER2013 (`data/fer2013.csv`)
