# AI Lab Task #02: Complete Workflow - Quick Reference

## Project Structure
```
AI-Cinematic-Impact-Synthesizer/
├── README.md                          # Project overview
├── SETUP_GUIDE.md                     # Task #02 setup & training
├── TRAILER_GUIDE.md                   # Creepy trailer generation
├── task2_cinematic_impact.py           # Train impact classification model
├── generate_creepy_trailer.py          # Generate horror trailer
├── requirements.txt                    # Python dependencies
├── data/                               # Input videos (0.mp4 - 18.mp4)
└── output/                             # Generated files
    ├── impact_model.pkl                # Trained model
    ├── scaler.pkl                      # Feature scaler
    ├── training_summary.txt            # Training metrics
    ├── creepy_trailer.mp4              # Final horror trailer
    └── impact_timeline.png             # Impact score visualization
```

## Phase 1: Model Training

### Command
```bash
python task2_cinematic_impact.py
```

### What It Does
1. **Extracts Features** (per video):
   - Motion Intensity (Optical Flow)
   - Object Presence (YOLO11 detection)
   - CNN Embeddings (ResNet18)
   
2. **Auto-Labels Data**:
   - High-impact (+1) if motion > 15.0 OR area > 5000.0
   - Low-impact (-1) otherwise
   
3. **Trains Model**:
   - Logistic Regression with L2 regularization
   - 70/30 train/test split
   
4. **Saves Artifacts**:
   - impact_model.pkl
   - scaler.pkl
   - training_summary.txt

### Output
- Model accuracy score
- Classification metrics
- Training summary

### Time
- ~3-10 min (CPU) | ~1-2 min (GPU)

---

## Phase 2: Creepy Trailer Generation

### Command
```bash
python generate_creepy_trailer.py
```

### What It Does
1. **Scores Video Segments**:
   - Uses trained impact model
   - Evaluates all segments with 50% overlap
   
2. **Selects Top 5 Clips**:
   - Highest impact scores
   - From different source videos
   
3. **Applies Visual Effects**:
   - Shadow-Silhouette (darken character to 30%)
   - Glowing Red Eyes (red glow effect)
   - Background Desaturation (20% color)
   
4. **Adds Captions**:
   - Random spooky text overlays
   - Red text on dark background
   - Bottom center positioning
   
5. **Generates Outputs**:
   - creepy_trailer.mp4 (5 clips, ~25 sec)
   - impact_timeline.png (visualization)

### Output
- Horror trailer video
- Impact score timeline chart

### Time
- ~10-20 min (CPU) | ~3-5 min (GPU)

---

## Feature Dimensions

### Total Feature Vector: 515 dimensions

| Component | Dimensions | Description |
|-----------|-----------|-------------|
| Motion Intensity | 1 | Average optical flow magnitude |
| Object Count | 1 | Avg characters per frame |
| Object Area | 1 | Avg bounding box area |
| CNN Embeddings | 512 | ResNet18 features |

---

## Configuration Quick Reference

### Motion & Object Thresholds
```python
'motion_threshold': 15.0        # Optical flow magnitude
'area_threshold': 5000.0        # Bounding box area
```

### Visual Effects
```python
'darkening_factor': 0.3         # Shadow intensity (0-1)
'background_saturation': 0.2    # Color retention (0-1)
'red_eye_glow_radius': 3        # Eye glow size
```

### Trailer Settings
```python
'num_clips': 5                   # Clips in final trailer
'clip_duration': 5               # Seconds per clip
'fps': 24                        # Video framerate
```

### Text Overlay
```python
'font_size': 24                  # Caption text size
'text_color': (0, 0, 255)        # Red in BGR
'text_alpha': 0.9                # Opacity
```

---

## Installation Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. GPU Setup (Optional but Recommended)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Verify Setup
```bash
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

---

## Common Workflows

### Train From Scratch
```bash
python task2_cinematic_impact.py          # Step 1
python generate_creepy_trailer.py         # Step 2
```

### Retrain with Different Thresholds
```python
# Edit task2_cinematic_impact.py
CONFIG['motion_threshold'] = 20.0         # More strict
CONFIG['area_threshold'] = 7000.0

python task2_cinematic_impact.py          # Retrain
python generate_creepy_trailer.py         # Generate new trailer
```

### Generate New Trailer (Same Model)
```bash
python generate_creepy_trailer.py         # Uses existing model
```

### Generate Trailer with Fewer Clips
```python
# Edit generate_creepy_trailer.py
CONFIG['num_clips'] = 3                   # 3 clips instead of 5

python generate_creepy_trailer.py
```

### Make Creepier Effects
```python
# Edit generate_creepy_trailer.py
CONFIG['darkening_factor'] = 0.5          # Darker shadows
CONFIG['background_saturation'] = 0.1     # More gray
CONFIG['font_size'] = 32                  # Bigger text

python generate_creepy_trailer.py
```

---

## Troubleshooting Checklist

| Issue | Solution |
|-------|----------|
| Videos not found | Check `./data/` folder, use .mp4/.avi/.mov formats |
| YOLO error | Install: `pip install ultralytics` |
| Model not found | Run training first: `python task2_cinematic_impact.py` |
| Out of memory | Use GPU or reduce `clip_duration` |
| Video won't play | Try different codec in CONFIG |
| Slow processing | Use GPU: `pip install torch[cuda118]` |

---

## Expected Results Summary

### Training Phase
- Accuracy: 70-85% (varies by data quality)
- Feature extraction: 515-dim vectors
- Model size: ~50KB
- Training time: 1-10 minutes

### Trailer Phase
- Duration: ~25 seconds (5 clips × 5 sec)
- File size: 5-50MB (depends on resolution)
- Visual effects: 3 layers applied
- Captions: 5 random spooky texts
- Generation time: 5-20 minutes

---

## Advanced Tips

### Use Custom Captions
Edit `CREEPY_PROMPTS` in `generate_creepy_trailer.py`:
```python
CREEPY_PROMPTS = {
    'character': [
        'YOUR CUSTOM TEXT HERE...',
        'MORE SCARY STUFF...',
    ],
    'motion': [...],
    'end': [...]
}
```

### Adjust Eye Position
```python
# In add_glowing_red_eyes()
eye_y = int(y1 + box_height * 0.35)    # Adjust 0.3 to 0.35
```

### Add Audio
```bash
ffmpeg -i creepy_trailer.mp4 -i audio.mp3 -c:v copy -shortest output.mp4
```

### Batch Process Videos
Modify scripts to handle multiple video folders

---

## Model Details

### Algorithm
- Logistic Regression
- L2 Regularization (Ridge)
- LBFGS Solver
- Max Iterations: 1000

### Training Data
- 19 videos
- 515-dimensional features
- 70/30 train/test split
- Stratified split (preserve label distribution)

### Input Requirements
- Video format: MP4, AVI, MOV
- No audio required (visual features only)
- Any resolution supported

### Output Interpretation
- Prediction: +1 (High-impact) or -1 (Low-impact)
- Probability: 0.0-1.0 confidence score

---

## Performance Benchmarks

| Device | Training Time | Trailer Gen Time | Total |
|--------|---------------|------------------|-------|
| CPU (Intel i5) | 8-10 min | 15-20 min | 25-30 min |
| CPU (Intel i7) | 5-7 min | 10-15 min | 15-22 min |
| GPU (GTX 1080) | 1-2 min | 3-5 min | 4-7 min |
| GPU (RTX 3090) | 30-60 sec | 2-3 min | 2.5-4 min |

---

## Next Steps After Generation

1. ✅ Review `creepy_trailer.mp4` video
2. ✅ Check `impact_timeline.png` visualization
3. ✅ Verify `training_summary.txt` metrics
4. 🎬 Share/present the trailer
5. 📊 Analyze results and iterate

---

## Help & Support

### View Detailed Guides
- **Training**: See `SETUP_GUIDE.md`
- **Trailer**: See `TRAILER_GUIDE.md`

### Check Script Docstrings
```bash
python -c "import task2_cinematic_impact; help(task2_cinematic_impact.extract_motion_intensity)"
```

### Debug Output
Uncomment `verbose=True` in YOLO calls for more details

