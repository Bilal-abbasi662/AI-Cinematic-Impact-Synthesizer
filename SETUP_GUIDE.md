# Task #02: Cinematic Impact Classification Model - Setup Guide

## Overview
This script builds a machine learning classification model to predict cinematic impact (High-impact vs Low-impact) based solely on visual features extracted from video clips.

## Prerequisites
- Python 3.8 or higher
- CUDA 11.8+ (optional, for GPU acceleration - highly recommended for faster processing)
- 4GB+ RAM minimum

## Installation

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. GPU Support (Recommended)
If you have an NVIDIA GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Download YOLO11 Model
The script will automatically download the YOLO11 nano model on first run. Make sure you have internet access.

## Project Structure
```
AI-Cinematic-Impact-Synthesizer/
├── task2_cinematic_impact.py    # Main script
├── requirements.txt              # Python dependencies
├── data/                         # Video files (0.mp4 - 18.mp4)
├── output/                       # Generated files
│   ├── impact_model.pkl         # Trained model
│   ├── scaler.pkl               # Feature scaler
│   └── training_summary.txt     # Training results
└── README.md
```

## Usage

### Running the Script
```bash
python task2_cinematic_impact.py
```

### Expected Output
The script will:
1. **Load Models** - Initialize YOLO11 and ResNet18
2. **Extract Features** - Process all videos and extract:
   - Motion intensity (optical flow)
   - Object presence (YOLO detection)
   - CNN embeddings (ResNet18)
3. **Auto-Label Data** - Create labels based on thresholds
4. **Train Model** - Train Logistic Regression classifier
5. **Save Artifacts** - Export model and scaler

### Processing Time
- **Per video**: ~10-30 seconds (CPU), ~2-5 seconds (GPU)
- **Total (19 videos)**: ~3-10 minutes (CPU), ~1-2 minutes (GPU)

## Feature Extraction Details

### 1. Motion Intensity
- **Method**: OpenCV Optical Flow (Farneback algorithm)
- **Calculation**: Average magnitude of motion vectors per frame
- **Output**: Single scalar value

### 2. Object Presence
- **Method**: YOLO11 nano model
- **Target**: 'stitchpunk' characters (Class 0)
- **Metrics**: 
  - Average character count per sampled frame
  - Average bounding box area per sampled frame
- **Output**: 2 scalar values

### 3. CNN Embeddings
- **Model**: ResNet18 (pre-trained on ImageNet)
- **Feature Dimension**: 512
- **Sampling**: Every 10th frame
- **Output**: 512 scalar values

**Total Feature Vector**: 515 dimensions (1 + 2 + 512)

## Model Configuration

### Thresholds
- **Motion Threshold**: 15.0 (average optical flow magnitude)
- **Area Threshold**: 5000.0 (bounding box area)
- A video is labeled as **High-impact** if EITHER:
  - Motion intensity > 15.0, OR
  - Object area > 5000.0

### Model Parameters
- **Algorithm**: Logistic Regression with L2 regularization
- **Regularization**: L2 penalty (ridge regression)
- **Train/Test Split**: 70/30
- **Max Iterations**: 1000
- **Solver**: LBFGS

## Output Files

### 1. `impact_model.pkl`
- Trained Logistic Regression model
- Use for predictions on new videos

### 2. `scaler.pkl`
- Feature standardization scaler
- Required for preprocessing new data

### 3. `training_summary.txt`
- Training statistics and results
- Model performance metrics

## Using the Trained Model

```python
import pickle
import numpy as np

# Load model and scaler
with open('output/impact_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('output/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare features (extract from new video)
features = np.array([...])  # 515-dimensional feature vector

# Normalize and predict
features_scaled = scaler.transform([features])
prediction = model.predict(features_scaled)  # Returns: 1 (High-impact) or -1 (Low-impact)
confidence = model.predict_proba(features_scaled)  # Get confidence scores
```

## Troubleshooting

### Issue: "CUDA out of memory"
- **Solution**: CPU processing is automatically used if GPU runs out of memory

### Issue: YOLO model not downloading
- **Solution**: Manually download: `yolo detect predict model=yolo11n.pt source=test.mp4`

### Issue: Video not opening
- **Solution**: Ensure videos are in `./data/` folder and use supported formats (.mp4, .avi, .mov)

### Issue: Process taking too long
- **Solution**: Use GPU (requires CUDA), or reduce number of videos for testing

## Customization

Edit the `CONFIG` dictionary in `task2_cinematic_impact.py`:

```python
CONFIG = {
    'motion_threshold': 15.0,      # Increase for stricter High-impact criteria
    'area_threshold': 5000.0,      # Adjust object size threshold
    'sample_frame_interval': 10,   # Extract features every Nth frame
    'test_size': 0.30,             # Train/test split ratio
}
```

## Performance Metrics

The script outputs:
- **Accuracy**: Overall classification accuracy on test set
- **Precision/Recall**: For both High-impact and Low-impact classes
- **Confusion Matrix**: Shows true positives, false positives, etc.

## Next Steps

1. Run the script on your 19 videos
2. Review `training_summary.txt` for model performance
3. Use `impact_model.pkl` and `scaler.pkl` for new predictions
4. Adjust thresholds if needed and retrain
