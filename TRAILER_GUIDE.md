# Creepy Trailer Generator - Usage Guide

## Overview
This script generates a horror-themed "creepy trailer" by:
1. Scoring all video segments using your trained impact model
2. Selecting the 5 highest-scoring segments from different videos
3. Applying visual horror effects (darkening, red glowing eyes, background desaturation)
4. Generating and overlaying spooky captions
5. Compiling into a final trailer with impact score visualization

## Prerequisites
- Trained `impact_model.pkl` from Task #02
- Scaler `scaler.pkl` from Task #02
- Videos in `./data/` folder
- All dependencies from requirements.txt installed

## Running the Script

```bash
python generate_creepy_trailer.py
```

### Processing Time
- **Per video**: 30-60 seconds (varies by length)
- **Total (19 videos)**: 10-20 minutes (CPU), 3-5 minutes (GPU)

## Output Files

### 1. `creepy_trailer.mp4`
- Final horror trailer video
- 5 clips from different videos
- Duration: ~25 seconds (5 clips × 5 seconds each)
- Resolution: Original video resolution
- Features:
  - Darkened character silhouettes
  - Glowing red eyes effect
  - Desaturated backgrounds (20% color)
  - Spooky animated captions

### 2. `impact_timeline.png`
- Two-panel visualization:
  - **Top**: Bar chart showing impact scores of 5 selected clips
  - **Bottom**: Timeline showing impact score progression through trailer

## Visual Effects Explained

### Shadow-Silhouette Effect
- Darkens character ROI to 30% of original brightness
- Creates ominous shadow appearance
- Applied per detected character

### Glowing Red Eyes
- Replaces eyes with bright red glow (BGR: 0, 0, 255)
- Glow radius: 3 pixels
- Eyes positioned in upper third of character bounding box
- Creates supernatural horror effect

### Background Desaturation
- Reduces background colors to 20% saturation
- Creates bleak, washed-out environment
- Keeps character ROI relatively more vivid
- Increases contrast and creepiness

### Caption Overlay
- 5 random spooky captions per clip
- Captions fade in/out throughout video
- Positioned at bottom center of frame
- Semi-transparent black background with red text

## Creepy Caption Types

### Character-Focused
- "The shadow awakens..."
- "It watches from the dark..."
- "They are never alone..."
- "The eyes in the dark see you..."

### Motion-Focused
- "Movement in the darkness..."
- "Something is coming..."
- "The air grows cold..."
- "An unseen force approaches..."

### Finale
- "YOUR END APPROACHES"
- "RESISTANCE IS FUTILE"
- "THE RECKONING BEGINS"
- "OBLIVION AWAITS"

## Configuration

Edit `CONFIG` dictionary in `generate_creepy_trailer.py`:

```python
CONFIG = {
    'num_clips': 5,                    # Number of clips in trailer
    'clip_duration': 5,                # Seconds per clip
    'fps': 24,                         # Video frame rate
    'darkening_factor': 0.3,           # How dark shadows are (0-1)
    'background_saturation': 0.2,      # Background color retention (0-1)
    'font_size': 24,                   # Caption text size
    'text_alpha': 0.9,                 # Caption opacity (0-1)
}
```

### Customization Ideas
- Increase `darkening_factor` to 0.5 for more dramatic shadows
- Set `background_saturation` to 0.0 for black & white effect
- Reduce `font_size` to 18 for smaller captions
- Add custom captions to `CREEPY_PROMPTS` dictionary

## Algorithm Details

### Clip Selection
1. Processes all video segments (with 50% overlap)
2. Extracts same features as training (motion, objects, CNN embeddings)
3. Scores each segment using trained impact model
4. Selects top 5 high-impact predictions
5. Ensures clips come from different source videos

### Impact Score Calculation
- Uses model's probability output for high-impact class
- Higher scores = higher likelihood of high-impact content
- Range: 0.0 to 1.0

### Feature Extraction for Scoring
Same pipeline as training:
- **Motion Intensity**: Optical flow average magnitude
- **Object Presence**: YOLO11 character detection count & area
- **CNN Features**: ResNet18 512-dimensional embeddings

## Troubleshooting

### Error: "Model not found"
```
Make sure impact_model.pkl and scaler.pkl exist in ./output/
Run task2_cinematic_impact.py first to train the model
```

### Error: "YOLO model failed"
```
Install: pip install ultralytics
Or update: pip install --upgrade ultralytics
```

### Error: "No high-impact clips found"
```
Adjust thresholds in task2_cinematic_impact.py:
  - Increase motion_threshold or area_threshold
  - Retrain the model
```

### Video codec issues
```
If trailer won't play, change video_codec:
  - 'mp4v' (default)
  - 'H264'
  - 'DIVX'
```

### Out of memory
```
Reduce clip_duration from 5 to 3
Or process fewer videos at a time
GPU memory can be freed between clips
```

## Quality Tips

### For More Creepy Effect
- Increase `darkening_factor` to 0.4-0.5
- Set `background_saturation` to 0.1-0.15
- Use larger font sizes for text
- Add more captions

### For Better Video Quality
- Keep original video resolution
- Use `fps`: 30 for smoother motion
- Increase clip duration for longer trailers

### For Scarier Text
- Edit `CREEPY_PROMPTS` with custom spooky text
- Use ALL CAPS for emphasis
- Add ellipses "..." for suspense

## Advanced Usage

### Custom Caption Generator
Replace `generate_creepy_caption()` with API-based system:

```python
def generate_creepy_caption(caption_type='character'):
    # Use OpenAI, Hugging Face, or other NLP service
    response = nlp_model.generate(prompt=f"Create a creepy caption about {caption_type}")
    return response
```

### Variable Speed Playback
Add frame repetition for slow-motion horror effect:

```python
# In process_creepy_segment()
for frame in frames:
    for _ in range(2):  # Double each frame
        processed_frames.append(frame_effects)
```

### Add Audio
After video generation:
```bash
ffmpeg -i creepy_trailer.mp4 -i horror_music.mp3 -c:v copy -c:a aac -shortest output.mp4
```

## Expected Output Example

```
==================================================================
🎬 CREEPY TRAILER GENERATOR
==================================================================

✓ Output directory ready: ./output
✓ Model and scaler loaded
✓ ResNet18 model loaded (device: cuda)
✓ YOLO11 model loaded

📹 Found 19 videos

🎯 Scoring video segments...
Scoring videos: 100%|████████| 19/19

✓ Selected 5 high-impact clips
  Clip 1: 0.mp4 - Score: 0.8945
  Clip 2: 3.mp4 - Score: 0.8723
  Clip 3: 7.mp4 - Score: 0.8612
  Clip 4: 11.mp4 - Score: 0.8501
  Clip 5: 15.mp4 - Score: 0.8234

✨ Applying creepy visual effects...
Processing clips: 100%|████████| 5/5

🎞️  Creating trailer video...
Writing video: 100%|████████| 600/600

📊 Generating impact timeline...
✓ Timeline plot saved: ./output/impact_timeline.png

==================================================================
✅ CREEPY TRAILER COMPLETE!
==================================================================
🎬 Trailer video: ./output/creepy_trailer.mp4
📈 Timeline plot: ./output/impact_timeline.png
⏱️  Duration: 25.00 seconds
==================================================================
```

## Next Steps

1. Run the script and check `creepy_trailer.mp4`
2. Review `impact_timeline.png` visualization
3. Adjust parameters if needed
4. Share the spooky trailer!

