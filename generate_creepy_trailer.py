"""
Creepy Trailer Generator
Generates a horror-themed trailer by selecting high-impact clips and applying
visual/audio effects with captions.

Pipeline:
1. Score all clips using the trained impact model
2. Select top 5 High-impact clips from different videos
3. Apply visual transformations: shadow-silhouette, glowing red eyes, desaturation
4. Generate and overlay creepy captions
5. Compile final trailer and generate impact score timeline
"""

import os
import pickle
import sys
import cv2
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
from pathlib import Path
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image, ImageDraw, ImageFont
import warnings
from tqdm import tqdm
from collections import defaultdict
import re

warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_dir': './data',
    'output_dir': './output',
    'model_path': './output/impact_model.pkl',
    'scaler_path': './output/scaler.pkl',
    'trailer_output': './output/creepy_trailer.mp4',
    'timeline_plot': './output/impact_timeline.png',
    'yolo_model': 'yolo11n.pt',
    'allow_model_downloads': False,  # set True to allow model weight downloads
    
    # Trailer settings
    'clip_duration': 4,  # seconds per clip (3-4 seconds allowed by spec)
    'transition_duration': 1,  # seconds of black glitch transition
    'target_duration': 40,  # total trailer duration in seconds
    'video_reuse_cooldown_unique': 10,  # require this many unique videos before reusing one
    'fps': 24,
    'video_codec': 'mp4v',
    'yolo_confidence': 0.5,
    
    # Visual effects
    'darkening_factor': 0.25,  # How much to darken character ROI
    'red_eye_color': (0, 0, 255),  # BGR format
    'red_eye_glow_radius': 6,
    'background_saturation': 0.28,  # Higher than before to avoid overly dull output
    'global_saturation_boost': 1.12,
    'global_brightness_boost': 1.06,
    
    # Text overlay
    'font_size': 28,
    'text_color': (0, 0, 255),  # Red for creepy effect
    'text_alpha': 1.0,
    'text_position': 'bottom_center',
    'text_bg_color': (0, 0, 0),
    'text_bg_alpha': 0.7,
    'font_candidates': [
        'Cinzel-Regular.ttf',
        'BebasNeue-Regular.ttf',
        'TrajanPro-Regular.ttf',
        'timesbd.ttf',
        'arial.ttf',
    ],
}

# ============================================================================
# CREEPY TEXT MAPPING
# ============================================================================

CREEPY_PROMPTS = {
    'character': [
        'Fabric and fear...',
        'Stitched in silence...',
        'Threaded with dread...',
        'The shadow awakens...',
        'It watches from the dark...',
        'Something stirs in silence...',
        'They are never alone...',
        'The eyes in the dark see you...',
        'A presence lingers...',
        'Something moves in the void...',
        'The darkness whispers your name...',
        'An unseen force approaches...',
        'The silence is watching...',
    ],
    'motion': [
        'Movement in the darkness...',
        'Something is coming...',
        'The air grows cold...',
        'Reality bends here...',
        'An unknown force approaches...',
        'The veil grows thin...',
        'It draws closer...',
        'The void expands...',
        'Shadows dance with hunger...',
        'Something ancient stirs...',
    ],
    'end': [
        'YOUR END APPROACHES',
        'RESISTANCE IS FUTILE',
        'THE RECKONING BEGINS',
        'FEAR CONSUMES ALL',
        'NO ESCAPE AWAITS',
        'DARKNESS CLAIMS SOULS',
        'THE HUNT BEGINS',
        'OBLIVION AWAITS',
    ]
}

# ============================================================================
# SETUP
# ============================================================================

def setup_directories():
    """Create output directories if they don't exist."""
    Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)

def load_model_and_scaler():
    """Load trained model and feature scaler."""
    with open(CONFIG['model_path'], 'rb') as f:
        model = pickle.load(f)
    
    with open(CONFIG['scaler_path'], 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"✓ Model and scaler loaded")
    return model, scaler

def load_resnet_model():
    """Load pre-trained ResNet18 for feature extraction."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_dir = Path(CONFIG['output_dir']) / 'torch_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(cache_dir))

    try:
        if hasattr(models, 'ResNet18_Weights'):
            weights = models.ResNet18_Weights.DEFAULT if CONFIG.get('allow_model_downloads', False) else None
            backbone = models.resnet18(weights=weights)
        else:
            backbone = models.resnet18(pretrained=CONFIG.get('allow_model_downloads', False))
    except Exception:
        if hasattr(models, 'ResNet18_Weights'):
            backbone = models.resnet18(weights=None)
        else:
            backbone = models.resnet18(pretrained=False)

    model = torch.nn.Sequential(*list(backbone.children())[:-1])
    model.eval()
    model = model.to(device)

    return model, device

def load_yolo_model():
    """Load YOLO model with graceful fallback."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[WARN] YOLO not installed. Install with: pip install ultralytics")
        return None

    candidates = [CONFIG.get('yolo_model', 'yolo11n.pt'), 'yolo11n.pt', 'yolov8n.pt']
    seen = set()

    for weight in candidates:
        if weight in seen:
            continue
        seen.add(weight)

        if (not CONFIG.get('allow_model_downloads', False)) and (not Path(weight).is_file()):
            print(f"[WARN] Skipping '{weight}' (not found locally; downloads disabled).")
            continue

        try:
            model = YOLO(weight)
            print(f"[OK] YOLO model loaded: {weight}")
            return model
        except Exception as e:
            print(f"[WARN] Could not load YOLO model '{weight}': {e}")

    print("[WARN] Continuing without YOLO detections.")
    return None

# ============================================================================
# FEATURE EXTRACTION (SAME AS TRAINING)
# ============================================================================

def extract_motion_intensity(video_path, start_frame=0, end_frame=None, sample_interval=10):
    """Extract motion intensity for a specific frame range."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return 0.0
    
    motion_values = []
    prev_frame = None
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if end_frame and frame_count >= end_frame:
                break
            
            if frame_count >= start_frame and frame_count % sample_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    motion_values.append(np.mean(mag))
                
                prev_frame = gray
            
            frame_count += 1
    finally:
        cap.release()
    
    return np.mean(motion_values) if motion_values else 0.0

def extract_object_presence(video_path, yolo_model, start_frame=0, end_frame=None, sample_interval=10):
    """Extract object presence for a specific frame range."""
    if yolo_model is None:
        return 0, 0.0
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return 0, 0.0
    
    total_characters = 0
    total_area = 0.0
    frame_count = 0
    sampled_frames = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if end_frame and frame_count >= end_frame:
                break
            
            if frame_count >= start_frame and frame_count % sample_interval == 0:
                sampled_frames += 1
                
                results = yolo_model(frame, verbose=False)
                
                for result in results:
                    if result.boxes is not None:
                        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                            if int(cls.item()) == 0:
                                total_characters += 1
                                x1, y1, x2, y2 = box.cpu().numpy()
                                area = (x2 - x1) * (y2 - y1)
                                total_area += area
            
            frame_count += 1
    finally:
        cap.release()
    
    avg_characters = total_characters / sampled_frames if sampled_frames > 0 else 0
    avg_area = total_area / sampled_frames if sampled_frames > 0 else 0.0
    
    return avg_characters, avg_area

def extract_cnn_embeddings(video_path, resnet_model, device, start_frame=0, end_frame=None, sample_interval=10):
    """Extract CNN embeddings for a specific frame range."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return np.zeros(512)
    
    embeddings = []
    frame_count = 0
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    try:
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if end_frame and frame_count >= end_frame:
                    break
                
                if frame_count >= start_frame and frame_count % sample_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    
                    frame_tensor = transform(frame_pil).unsqueeze(0).to(device)
                    embedding = resnet_model(frame_tensor)
                    embeddings.append(embedding.cpu().numpy().flatten())
                
                frame_count += 1
    finally:
        cap.release()
    
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(512)

def score_video_segments(video_path, yolo_model, resnet_model, device, scaler, model):
    """
    Score all segments of a video to find highest-impact clips.
    
    Returns:
        List of (start_frame, end_frame, impact_score, prediction)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    segment_length = int(fps * CONFIG['clip_duration'])  # Frames per clip
    segments = []
    
    for start in range(0, frame_count - segment_length, segment_length // 2):  # 50% overlap
        end = start + segment_length
        
        # Extract features
        motion = extract_motion_intensity(video_path, start, end)
        char_count, area = extract_object_presence(video_path, yolo_model, start, end)
        cnn_embed = extract_cnn_embeddings(video_path, resnet_model, device, start, end)
        
        # Combine features
        features = np.concatenate([[motion, char_count, area], cnn_embed])
        features_scaled = scaler.transform([features])
        
        # Get prediction and probability
        pred = int(model.predict(features_scaled)[0])
        prob = model.predict_proba(features_scaled)[0]

        # Score by probability of the high-impact class (+1), robust to class ordering.
        classes = list(model.classes_)
        if 1 in classes:
            impact_score = float(prob[classes.index(1)])
        else:
            impact_score = float(np.max(prob))
        
        segments.append({
            'start': start,
            'end': end,
            'score': impact_score,
            'prediction': pred
        })
    
    return sorted(segments, key=lambda x: x['score'], reverse=True)

def required_segment_count(target_duration, clip_duration, transition_duration):
    """
    Compute the number of segments needed before tail padding.
    For n segments and (n-1) transitions: n*clip + (n-1)*transition <= target.
    """
    if clip_duration <= 0:
        return 1

    n = 1
    while (n * clip_duration + max(0, n - 1) * transition_duration) <= target_duration:
        n += 1
    return max(1, n - 1)

def build_video_blacklist(selection_history, cooldown_unique):
    """
    Blacklist videos that were used recently and have not yet been separated by
    at least `cooldown_unique` other unique videos.
    """
    blacklist = set()
    for video_id in set(selection_history):
        last_idx = len(selection_history) - 1 - selection_history[::-1].index(video_id)
        unique_since = set(selection_history[last_idx + 1:])
        if len(unique_since) < cooldown_unique:
            blacklist.add(video_id)
    return blacklist

def select_diverse_segments(all_clips):
    """
    Select high-score segments with strict video uniqueness cooldown.
    A reused video is blocked until at least N other unique videos are used.
    """
    ranked = sorted(all_clips, key=lambda x: x['score'], reverse=True)
    target_segments = required_segment_count(
        CONFIG['target_duration'],
        CONFIG['clip_duration'],
        CONFIG['transition_duration']
    )
    cooldown_unique = int(CONFIG.get('video_reuse_cooldown_unique', 10))

    selected = []
    selection_history = []
    used_keys = set()

    while len(selected) < target_segments:
        blacklist = build_video_blacklist(selection_history, cooldown_unique)
        picked = None

        for clip in ranked:
            key = (clip['video'], clip['start'], clip['end'])
            if key in used_keys:
                continue

            video_id = clip['video_id']
            if video_id in blacklist:
                continue

            picked = clip
            break

        if picked is None:
            break

        selected.append(picked)
        selection_history.append(picked['video_id'])
        used_keys.add((picked['video'], picked['start'], picked['end']))

    return selected

# ============================================================================
# VISUAL EFFECTS
# ============================================================================

def _clamp_bbox(bbox, width, height):
    """Clamp bbox to frame boundaries."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2

def enhance_texture_roi(roi):
    """Highlight fabric and mechanical textures in ROI using CLAHE + detailEnhance."""
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_chan)
    roi_clahe = cv2.cvtColor(cv2.merge([l_enhanced, a_chan, b_chan]), cv2.COLOR_LAB2BGR)
    return cv2.detailEnhance(roi_clahe, sigma_s=10, sigma_r=0.15)

def darken_face_region(roi, darkening_factor=0.3):
    """Darken upper-face area inside ROI for a more ominous look."""
    h, w = roi.shape[:2]
    if h < 6 or w < 6:
        return roi
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, int(h * 0.30))
    axes = (max(2, int(w * 0.30)), max(2, int(h * 0.18)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    darkened = roi.copy()
    factor = float(np.clip(1.0 - darkening_factor, 0.2, 1.0))
    darkened[mask > 0] = (darkened[mask > 0].astype(np.float32) * factor).astype(np.uint8)
    return darkened

def detect_eye_points(roi):
    """Detect likely eye points from bright components in upper ROI."""
    h, w = roi.shape[:2]
    if h < 10 or w < 10:
        return [(int(w * 0.35), int(h * 0.30)), (int(w * 0.65), int(h * 0.30))]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    upper = gray[:max(1, int(h * 0.55)), :]
    blur = cv2.GaussianBlur(upper, (5, 5), 0)
    threshold = max(150, int(np.percentile(blur, 92)))
    _, mask = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 2:
            continue
        m = cv2.moments(cnt)
        if m['m00'] == 0:
            continue
        points.append((int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])))

    if len(points) >= 2:
        points = sorted(points, key=lambda p: p[0])
        return [points[0], points[-1]]

    return [(int(w * 0.35), int(h * 0.30)), (int(w * 0.65), int(h * 0.30))]

def add_glowing_red_eyes_roi(roi, eye_color=(0, 0, 255), glow_radius=3):
    """Add glowing red eyes directly within the ROI."""
    result = roi.copy()
    glow = np.zeros_like(roi)
    eye_points = detect_eye_points(roi)
    radius = max(3, glow_radius)

    for ex, ey in eye_points:
        ex = max(0, min(ex, roi.shape[1] - 1))
        ey = max(0, min(ey, roi.shape[0] - 1))
        cv2.circle(result, (ex, ey), radius + 1, eye_color, -1)
        cv2.circle(glow, (ex, ey), radius * 3, eye_color, -1)

    glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=4, sigmaY=4)
    out = cv2.addWeighted(result, 1.0, glow, 0.95, 0)
    out[:, :, 2] = np.clip(out[:, :, 2].astype(np.float32) * 1.10, 0, 255).astype(np.uint8)
    return out

def add_ambient_red_eyes(frame, eye_color=(0, 0, 255)):
    """
    Fallback red-eye effect when detector output is unavailable.
    Places subtle glowing eyes near upper-center of frame.
    """
    h, w = frame.shape[:2]
    cy = int(h * 0.33)
    sep = max(18, w // 14)
    radius = max(2, w // 220)
    left = (max(0, w // 2 - sep), cy)
    right = (min(w - 1, w // 2 + sep), cy)

    out = frame.copy()
    glow = np.zeros_like(frame)
    for ex, ey in [left, right]:
        cv2.circle(out, (ex, ey), radius + 1, eye_color, -1)
        cv2.circle(glow, (ex, ey), radius * 6, eye_color, -1)

    glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=6, sigmaY=6)
    return cv2.addWeighted(out, 1.0, glow, 0.45, 0)

def apply_creepy_effects(frame, yolo_model):
    """
    Apply all creepy visual effects to a frame.

    Args:
        frame: Input frame
        yolo_model: YOLO detection model

    Returns:
        Modified frame with effects applied
    """
    frame_copy = frame.copy()

    # Global grade to keep the trailer cinematic but not flat/dull.
    hsv_grade = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_grade[..., 1] = np.clip(hsv_grade[..., 1] * CONFIG.get('global_saturation_boost', 1.0), 0, 255)
    hsv_grade[..., 2] = np.clip(hsv_grade[..., 2] * CONFIG.get('global_brightness_boost', 1.0), 0, 255)
    graded = cv2.cvtColor(hsv_grade.astype(np.uint8), cv2.COLOR_HSV2BGR)
    graded = cv2.convertScaleAbs(graded, alpha=1.06, beta=4)

    # Start from a desaturated world, then paste enhanced character ROIs back.
    hsv_base = cv2.cvtColor(graded, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_base[..., 1] = hsv_base[..., 1] * CONFIG['background_saturation']
    base = cv2.cvtColor(np.clip(hsv_base, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    if yolo_model is None:
        return add_ambient_red_eyes(graded, CONFIG['red_eye_color'])

    try:
        results = yolo_model(frame_copy, verbose=False)
    except Exception:
        return graded

    h, w = frame_copy.shape[:2]
    found_box = False

    for result in results:
        if result.boxes is None:
            continue

        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            if int(cls.item()) != 0:
                continue
            if float(conf.item()) < CONFIG['yolo_confidence']:
                continue

            x1, y1, x2, y2 = _clamp_bbox(box.cpu().numpy(), w, h)
            if x2 <= x1 or y2 <= y1:
                continue

            found_box = True
            roi = graded[y1:y2, x1:x2].copy()
            roi = enhance_texture_roi(roi)
            roi = darken_face_region(roi, CONFIG['darkening_factor'])
            roi = add_glowing_red_eyes_roi(roi, CONFIG['red_eye_color'], CONFIG['red_eye_glow_radius'])
            base[y1:y2, x1:x2] = roi

    if not found_box:
        return add_ambient_red_eyes(graded, CONFIG['red_eye_color'])

    return base
# ============================================================================
# TEXT GENERATION AND OVERLAY
# ============================================================================

def generate_creepy_caption(caption_type='character'):
    """Generate a random creepy caption."""
    import random
    captions = CREEPY_PROMPTS.get(caption_type, CREEPY_PROMPTS['character'])
    return random.choice(captions)

def generate_segment_caption(clip, index):
    """Generate a cinematic creepy caption for a selected segment."""
    base_lines = [
        "A stitch in time, a soul in pain...",
        "Fabric and fear...",
        "Stitches remember everything...",
        "The seams are breathing...",
        "Rust whispers in the dark...",
        "No thread stays silent...",
        "The doll sees you...",
        "Steel nerves, stitched souls...",
        "Night is hand-stitched here...",
    ]
    return base_lines[index % len(base_lines)]

def load_cinematic_font(font_size):
    """Try cinematic fonts before falling back to default system fonts."""
    candidate_paths = []
    for name in CONFIG.get('font_candidates', []):
        candidate_paths.append(name)
        candidate_paths.append(os.path.join('fonts', name))
        candidate_paths.append(os.path.join('C:\\Windows\\Fonts', name))

    for font_path in candidate_paths:
        try:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, font_size)
        except Exception:
            pass

    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        return ImageFont.load_default()

def overlay_caption(frame, caption, position='bottom_center'):
    """
    Overlay text caption on frame.
    
    Args:
        frame: Input frame
        caption: Text to overlay
        position: 'top_center', 'bottom_center', etc.
    
    Returns:
        Frame with caption overlaid
    """
    frame_copy = frame.copy()
    h, w = frame.shape[:2]
    text_rgb = (CONFIG['text_color'][2], CONFIG['text_color'][1], CONFIG['text_color'][0])
    bg_rgb = (CONFIG['text_bg_color'][2], CONFIG['text_bg_color'][1], CONFIG['text_bg_color'][0])
    
    # Convert to PIL for better text rendering
    pil_image = Image.fromarray(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image, 'RGBA')
    
    font = load_cinematic_font(CONFIG['font_size'])
    
    # Get text bounding box to center it
    bbox = draw.textbbox((0, 0), caption, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate position
    if position == 'bottom_center':
        x = (w - text_width) // 2
        y = h - text_height - 20
    elif position == 'top_center':
        x = (w - text_width) // 2
        y = 20
    else:
        x = (w - text_width) // 2
        y = (h - text_height) // 2
    
    # Draw text background
    bg_padding = 10
    draw.rectangle(
        [x - bg_padding, y - bg_padding, x + text_width + bg_padding, y + text_height + bg_padding],
        fill=bg_rgb + (int(255 * CONFIG['text_bg_alpha']),)
    )

    # Draw a subtle red glow, then the main red text.
    glow_alpha = int(255 * min(1.0, CONFIG['text_alpha'] * 0.55))
    for ox, oy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        draw.text((x + ox, y + oy), caption, font=font, fill=text_rgb + (glow_alpha,))

    draw.text((x, y), caption, font=font, fill=text_rgb + (int(255 * CONFIG['text_alpha']),))
    
    # Convert back to OpenCV format
    frame_copy = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return frame_copy

# ============================================================================
# VIDEO PROCESSING
# ============================================================================

def extract_video_segment(video_path, start_frame, end_frame):
    """
    Extract frames from a video segment.
    
    Args:
        video_path: Path to video file
        start_frame: Starting frame number
        end_frame: Ending frame number
    
    Returns:
        List of frames, fps
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames = []
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count >= start_frame and frame_count < end_frame:
                frames.append(frame)
            
            if frame_count >= end_frame:
                break
            
            frame_count += 1
    finally:
        cap.release()
    
    return frames, fps

def process_creepy_segment(video_path, start_frame, end_frame, yolo_model, caption=None):
    """
    Extract video segment and apply creepy effects.
    
    Args:
        video_path: Path to video file
        start_frame: Starting frame number
        end_frame: Ending frame number
        yolo_model: YOLO detection model
    
    Returns:
        List of processed frames, fps
    """
    frames, fps = extract_video_segment(video_path, start_frame, end_frame)
    
    processed_frames = []
    
    for i, frame in enumerate(tqdm(frames, desc=f"Processing segment", leave=False)):
        # Apply creepy effects
        frame_effects = apply_creepy_effects(frame, yolo_model)

        # Keep caption visible for most of the segment.
        if caption and len(frames) > 0:
            start_show = int(len(frames) * 0.10)
            end_show = int(len(frames) * 0.90)
            if start_show <= i <= end_show:
                frame_effects = overlay_caption(frame_effects, caption)

        processed_frames.append(frame_effects)
    
    return processed_frames, fps

def create_glitch_transition(width, height, fps, duration_sec=1):
    """Create a black glitch transition."""
    frame_count = max(1, int(round(fps * duration_sec)))
    transition_frames = []

    for idx in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Sparse digital noise.
        noise_points = max(50, (width * height) // 2500)
        ys = np.random.randint(0, height, size=noise_points)
        xs = np.random.randint(0, width, size=noise_points)
        frame[ys, xs] = (np.random.randint(0, 40), np.random.randint(0, 40), np.random.randint(120, 255))

        # Horizontal glitch bars.
        if idx % 2 == 0:
            for _ in range(3):
                y = np.random.randint(0, max(1, height - 2))
                bar_h = np.random.randint(1, 3)
                color = (0, 0, np.random.randint(120, 220))
                frame[y:y + bar_h, :] = color

        transition_frames.append(frame)

    return transition_frames

def write_video(output_path, frames, fps, width, height):
    """
    Write frames to video file.
    
    Args:
        output_path: Output video file path
        frames: List of frames
        fps: Frames per second
        width: Frame width
        height: Frame height
    """
    fourcc = cv2.VideoWriter_fourcc(*CONFIG['video_codec'])
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in tqdm(frames, desc="Writing video"):
        out.write(frame)
    
    out.release()
    print(f"✓ Video saved: {output_path}")

# ============================================================================
# FINAL EVALUATION AND PLOTTING
# ============================================================================

def plot_impact_timeline(selected_clips, output_path):
    """
    Generate a plot showing impact scores over the trailer timeline.
    
    Args:
        selected_clips: List of selected clip dictionaries
        output_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Impact scores of selected clips
    clip_labels = [f"Clip {i+1}\n({os.path.basename(c['video'])})" for i, c in enumerate(selected_clips)]
    impact_scores = [c['score'] for c in selected_clips]
    colors = ['#FF0000' if s > 0.7 else '#FF6666' for s in impact_scores]
    
    ax1.bar(clip_labels, impact_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Impact Score', fontsize=12, fontweight='bold')
    ax1.set_title('Selected High-Impact Clips - Score Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Timeline of trailer with impact scores
    timeline_pos = np.arange(len(selected_clips))
    ax2.plot(timeline_pos, impact_scores, marker='o', linewidth=3, markersize=10, 
             color='#FF0000', markerfacecolor='#FF6666', markeredgewidth=2)
    ax2.fill_between(timeline_pos, impact_scores, alpha=0.3, color='#FF0000')
    ax2.set_xlabel('Clip Position in Trailer', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Impact Score', fontsize=12, fontweight='bold')
    ax2.set_title('Impact Score Timeline - Creepy Trailer Arc', fontsize=14, fontweight='bold')
    ax2.set_xticks(timeline_pos)
    ax2.set_xticklabels([f'Clip {i+1}' for i in range(len(selected_clips))])
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Timeline plot saved: {output_path}")
    plt.close()

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline."""
    print("=" * 70)
    print("🎬 CREEPY TRAILER GENERATOR")
    print("=" * 70)

    # Setup
    setup_directories()

    # Load models
    impact_model, scaler = load_model_and_scaler()
    resnet_model, device = load_resnet_model()
    yolo_model = load_yolo_model()

    if yolo_model is None:
        print("[WARN] YOLO unavailable; trailer will use fallback creepy effects without detections.")

    # Get video paths
    if not os.path.isdir(CONFIG['data_dir']):
        print(f"[ERROR] Data directory not found: {CONFIG['data_dir']}")
        return

    video_paths = sorted([
        os.path.join(CONFIG['data_dir'], f)
        for f in os.listdir(CONFIG['data_dir'])
        if f.lower().endswith(('.mp4', '.avi', '.mov'))
    ])

    if not video_paths:
        print(f"❌ No videos found in {CONFIG['data_dir']}")
        return

    print(f"\n📹 Found {len(video_paths)} videos")

    # Score all clips
    print("\n🎯 Scoring video segments...")
    all_clips = []

    for video_path in tqdm(video_paths, desc="Scoring videos"):
        segments = score_video_segments(video_path, yolo_model, resnet_model, device, scaler, impact_model)

        # Add video path to each segment
        video_id = os.path.basename(video_path)
        for segment in segments:
            segment['video'] = video_path
            segment['video_id'] = video_id
            all_clips.append(segment)

    if not all_clips:
        print("❌ No scorable segments found.")
        return

    # Rank globally and select with strict uniqueness cooldown (blacklist logic).
    selected_clips = select_diverse_segments(all_clips)
    planned_segments = required_segment_count(
        CONFIG['target_duration'],
        CONFIG['clip_duration'],
        CONFIG['transition_duration']
    )

    if len(selected_clips) < planned_segments:
        print(f"⚠ Only found {len(selected_clips)} segments under strict uniqueness rules.")
    if not selected_clips:
        print("❌ Could not select segments. Try lowering uniqueness cooldown or increasing data.")
        return

    print(f"\n✓ Selected {len(selected_clips)} high-impact clips")
    for i, clip in enumerate(selected_clips):
        print(f"  Clip {i+1}: {clip['video_id']} - Score: {clip['score']:.4f}")

    # Process selected clips
    print("\n✨ Applying creepy visual effects...")
    all_processed_frames = []
    fps_value = None

    for i, clip in enumerate(tqdm(selected_clips, desc="Processing clips")):
        segment_caption = generate_segment_caption(clip, i)
        frames, fps_value = process_creepy_segment(
            clip['video'],
            clip['start'],
            clip['end'],
            yolo_model,
            caption=segment_caption
        )
        all_processed_frames.extend(frames)

        if i < len(selected_clips) - 1 and frames:
            h, w = frames[0].shape[:2]
            next_video_id = selected_clips[i + 1]['video_id']
            if clip['video_id'] != next_video_id:
                all_processed_frames.extend(
                    create_glitch_transition(w, h, fps_value, CONFIG['transition_duration'])
                )

    # Write trailer video
    if all_processed_frames:
        h, w = all_processed_frames[0].shape[:2]

        # Enforce exact target duration by trimming/padding with glitch black.
        target_frames = int(round(CONFIG['target_duration'] * fps_value))
        if len(all_processed_frames) > target_frames:
            all_processed_frames = all_processed_frames[:target_frames]
        elif len(all_processed_frames) < target_frames:
            missing = target_frames - len(all_processed_frames)
            all_processed_frames.extend(
                create_glitch_transition(w, h, fps_value, missing / float(fps_value))
            )
            all_processed_frames = all_processed_frames[:target_frames]

        print("\n🎞️  Creating trailer video...")
        write_video(CONFIG['trailer_output'], all_processed_frames, fps_value, w, h)
    else:
        print("❌ No frames processed")
        return

    # Generate impact timeline plot
    print("\n📊 Generating impact timeline...")
    plot_impact_timeline(selected_clips, CONFIG['timeline_plot'])

    # Print summary
    print("\n" + "=" * 70)
    print("✅ CREEPY TRAILER COMPLETE!")
    print("=" * 70)
    print(f"🎬 Trailer video: {CONFIG['trailer_output']}")
    print(f"📈 Timeline plot: {CONFIG['timeline_plot']}")
    print(f"⏱️  Duration: {len(all_processed_frames) / fps_value:.2f} seconds")
    print("=" * 70)
if __name__ == "__main__":
    main()
