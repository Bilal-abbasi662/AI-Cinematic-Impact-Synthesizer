"""
Task #02: Cinematic Impact Classification Model
Builds a classification model based on visual features from video clips.

Features extracted:
- Motion Intensity: Using OpenCV Optical Flow
- Object Presence: YOLO11 'stitchpunk' character detection
- CNN Embeddings: ResNet18 512-dimensional features from every 10th frame

Model: Logistic Regression with L2 regularization
"""

import os
import pickle
import sys
import cv2
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
from tqdm import tqdm
from pathlib import Path

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
    'model_save_path': './output/impact_model.pkl',
    'scaler_save_path': './output/scaler.pkl',
    'video_extensions': ['.mp4', '.avi', '.mov'],
    
    # Feature extraction settings
    'sample_frame_interval': 10,  # Extract features every 10th frame
    'motion_threshold': 15.0,      # Motion intensity threshold for High-impact label
    'area_threshold': 5000.0,      # Bounding box area threshold for High-impact label
    'fallback_high_ratio': 0.30,   # used only if threshold labeling collapses to one class
    'fallback_min_class_samples': 2,
    
    # Model training settings
    'test_size': 0.30,
    'random_state': 42,
    'max_iterations': 1000,
    
    # YOLO settings
    'yolo_model': 'yolo11n.pt',    # nano model for efficiency
    'yolo_confidence': 0.5,
    'stitchpunk_class': 0,         # Class ID for 'stitchpunk' character
    'allow_model_downloads': False,  # set True to let PyTorch/YOLO download weights
}

# ============================================================================
# SETUP
# ============================================================================

def setup_directories():
    """Create output directories if they don't exist."""
    Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {CONFIG['output_dir']}")

def load_yolo_model():
    """Load YOLO model with graceful offline fallback."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[WARN] YOLO not installed. Install with: pip install ultralytics")
        return None

    candidates = [CONFIG['yolo_model'], 'yolo11n.pt', 'yolov8n.pt']
    seen = set()

    for weight in candidates:
        if weight in seen:
            continue
        seen.add(weight)

        if (not CONFIG.get('allow_model_downloads', False)) and (not Path(weight).is_file()):
            print(f"[WARN] Skipping '{weight}' (not found locally; downloads disabled).")
            continue

        try:
            print(f"[INFO] Loading YOLO model: {weight}...")
            model = YOLO(weight)
            print(f"[OK] YOLO model loaded: {weight}")
            return model
        except Exception as e:
            print(f"[WARN] Could not load YOLO model '{weight}': {e}")

    print("[WARN] Continuing without YOLO object features (character count/area set to 0).")
    return None
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
        if CONFIG.get('allow_model_downloads', False):
            print("[OK] Loaded pretrained ResNet18 weights")
        else:
            print("[WARN] Using ResNet18 without pretrained weights (downloads disabled).")
    except Exception as e:
        print(f"[WARN] Could not load pretrained ResNet18 weights: {e}")
        print("[WARN] Using randomly initialized ResNet18 as fallback.")
        if hasattr(models, 'ResNet18_Weights'):
            backbone = models.resnet18(weights=None)
        else:
            backbone = models.resnet18(pretrained=False)

    # Remove the classification layer to get embeddings
    model = torch.nn.Sequential(*list(backbone.children())[:-1])
    model.eval()

    # Move to GPU if available
    model = model.to(device)

    print(f"[OK] ResNet18 model loaded (device: {device})")
    return model, device

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_motion_intensity(video_path, sample_interval=10):
    """
    Calculate average motion intensity using Optical Flow.
    
    Args:
        video_path: Path to video file
        sample_interval: Process every Nth frame
        
    Returns:
        Average motion magnitude across sampled frames
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"⚠ Could not open video: {video_path}")
        return 0.0
    
    motion_values = []
    prev_frame = None
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % sample_interval != 0:
                continue
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            if prev_frame is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                # Calculate magnitude of motion vectors
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_values.append(np.mean(mag))
            
            prev_frame = gray
    finally:
        cap.release()
    
    return np.mean(motion_values) if motion_values else 0.0

def _clamp_bbox(x1, y1, x2, y2, width, height):
    """Clamp YOLO bbox coordinates to valid frame bounds."""
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))
    return x1, y1, x2, y2

def _detect_eye_points(roi):
    """
    Detect likely eye points inside a character ROI.
    Uses bright-point heuristics in upper ROI; falls back to symmetric anchors.
    """
    h, w = roi.shape[:2]
    if h < 10 or w < 10:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    upper_h = max(1, int(h * 0.55))
    upper = gray[:upper_h, :]
    blur = cv2.GaussianBlur(upper, (5, 5), 0)

    thresh_val = max(160, int(np.percentile(blur, 92)))
    _, mask = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2:
            continue
        m = cv2.moments(cnt)
        if m['m00'] == 0:
            continue
        cx = int(m['m10'] / m['m00'])
        cy = int(m['m01'] / m['m00'])
        candidates.append((area, cx, cy))

    # Choose two points with best left/right separation if detected.
    if len(candidates) >= 2:
        candidates = sorted(candidates, key=lambda x: x[0], reverse=True)[:6]
        best_pair = None
        best_sep = -1
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                _, x_a, y_a = candidates[i]
                _, x_b, y_b = candidates[j]
                sep = abs(x_a - x_b)
                if sep > best_sep:
                    best_sep = sep
                    best_pair = ((x_a, y_a), (x_b, y_b))
        if best_pair is not None:
            left, right = sorted(best_pair, key=lambda p: p[0])
            return [left, right]

    # Fallback anchors (upper-left / upper-right eye region).
    return [
        (int(w * 0.35), int(h * 0.30)),
        (int(w * 0.65), int(h * 0.30)),
    ]

def enhance_character_textures(frame, yolo_model):
    """
    Enhance detected character ROIs:
    - CLAHE texture isolation
    - detailEnhance for stitches/mechanical parts
    - glowing red eyes
    - 10% saturation for everything outside ROI
    """
    if yolo_model is None:
        return frame

    h, w = frame.shape[:2]
    # Desaturate full frame first, then paste enhanced ROIs back.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = hsv[..., 1] * 0.10
    output = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    try:
        results = yolo_model(frame, verbose=False)
    except Exception:
        return frame

    for result in results:
        if result.boxes is None:
            continue

        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            if int(cls.item()) != CONFIG['stitchpunk_class']:
                continue
            if float(conf.item()) < CONFIG['yolo_confidence']:
                continue

            x1, y1, x2, y2 = _clamp_bbox(*box.cpu().numpy(), w, h)
            if x2 <= x1 or y2 <= y1:
                continue

            roi = frame[y1:y2, x1:x2].copy()
            if roi.size == 0:
                continue

            # Texture isolation via CLAHE on luminance channel.
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_chan)
            roi_clahe = cv2.cvtColor(cv2.merge([l_enhanced, a_chan, b_chan]), cv2.COLOR_LAB2BGR)

            # Stitch/mechanical detail enhancement.
            roi_detail = cv2.detailEnhance(roi_clahe, sigma_s=10, sigma_r=0.15)

            # Creepy glowing red eyes inside ROI.
            eye_points = _detect_eye_points(roi_detail)
            glow = np.zeros_like(roi_detail)
            radius = max(2, min(roi_detail.shape[0], roi_detail.shape[1]) // 18)
            for ex, ey in eye_points:
                ex = max(0, min(ex, roi_detail.shape[1] - 1))
                ey = max(0, min(ey, roi_detail.shape[0] - 1))
                cv2.circle(roi_detail, (ex, ey), radius, (0, 0, 255), -1)
                cv2.circle(glow, (ex, ey), radius * 2, (0, 0, 255), -1)
            glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=3, sigmaY=3)
            roi_final = cv2.addWeighted(roi_detail, 1.0, glow, 0.7, 0)

            # Paste enhanced ROI over desaturated frame.
            output[y1:y2, x1:x2] = roi_final

    return output

def extract_object_presence(video_path, yolo_model, sample_interval=10):
    """
    Detect 'stitchpunk' characters and calculate their presence metrics.
    
    Args:
        video_path: Path to video file
        yolo_model: YOLO detection model
        sample_interval: Process every Nth frame
        
    Returns:
        Tuple: (character count, total bounding box area)
    """
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
            
            frame_count += 1
            if frame_count % sample_interval != 0:
                continue
            
            sampled_frames += 1
            
            # Run YOLO detection
            results = yolo_model(frame, verbose=False)
            
            for result in results:
                if result.boxes is not None:
                    for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                        # Check if it's a 'stitchpunk' character (class 0)
                        if int(cls.item()) == CONFIG['stitchpunk_class'] and float(conf.item()) >= CONFIG['yolo_confidence']:
                            total_characters += 1
                            # Calculate bounding box area
                            x1, y1, x2, y2 = box.cpu().numpy()
                            area = (x2 - x1) * (y2 - y1)
                            total_area += area
    finally:
        cap.release()
    
    # Normalize by number of sampled frames
    avg_characters = total_characters / sampled_frames if sampled_frames > 0 else 0
    avg_area = total_area / sampled_frames if sampled_frames > 0 else 0.0
    
    return avg_characters, avg_area

def extract_cnn_embeddings(video_path, resnet_model, device, yolo_model=None, sample_interval=10):
    """
    Extract CNN embeddings from ResNet18 for sampled frames.
    
    Args:
        video_path: Path to video file
        resnet_model: Pre-trained ResNet18 model
        device: Torch device (cuda/cpu)
        sample_interval: Process every Nth frame
        
    Returns:
        Average 512-dimensional embedding vector
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return np.zeros(512)
    
    embeddings = []
    frame_count = 0
    
    # Preprocessing transforms
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
                
                frame_count += 1
                if frame_count % sample_interval != 0:
                    continue
                
                # Enhance stitchpunk texture/eyes and isolate by desaturating background.
                enhanced_frame = enhance_character_textures(frame, yolo_model)

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
                from PIL import Image
                frame_pil = Image.fromarray(frame_rgb)
                
                # Apply transforms and extract embedding
                frame_tensor = transform(frame_pil).unsqueeze(0).to(device)
                embedding = resnet_model(frame_tensor)
                embeddings.append(embedding.cpu().numpy().flatten())
    finally:
        cap.release()
    
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(512)

def extract_all_features(video_paths, yolo_model, resnet_model, device):
    """
    Extract all features from video files.
    
    Args:
        video_paths: List of video file paths
        yolo_model: YOLO detection model
        resnet_model: ResNet18 model
        device: Torch device
        
    Returns:
        Features array (N x 515), list of video filenames
    """
    all_features = []
    video_names = []
    
    print("\n🎬 Extracting features from videos...")
    
    for video_path in tqdm(video_paths, desc="Feature extraction"):
        video_name = os.path.basename(video_path)
        
        # Extract motion intensity
        motion = extract_motion_intensity(video_path, CONFIG['sample_frame_interval'])
        
        # Extract object presence
        char_count, area = extract_object_presence(
            video_path, yolo_model, CONFIG['sample_frame_interval']
        )
        
        # Extract CNN embeddings
        cnn_embed = extract_cnn_embeddings(
            video_path, resnet_model, device, yolo_model, CONFIG['sample_frame_interval']
        )
        
        # Combine all features: [motion, char_count, area, ...cnn_embeddings(512)]
        features = np.concatenate([
            [motion, char_count, area],
            cnn_embed
        ])
        
        all_features.append(features)
        video_names.append(video_name)
    
    return np.array(all_features), video_names

# ============================================================================
# LABELING
# ============================================================================

def auto_label_data(features):
    """
    Auto-label frames as High-impact (+1) or Low-impact (-1).
    
    Logic: High-impact if motion intensity OR object area exceeds thresholds.
    
    Args:
        features: Feature array (N x 515)
        
    Returns:
        Labels array (N,)
    """
    labels = np.zeros(len(features))

    for i, feature in enumerate(features):
        motion = feature[0]
        area = feature[2]

        # High-impact if either motion or area exceeds threshold
        if motion > CONFIG['motion_threshold'] or area > CONFIG['area_threshold']:
            labels[i] = 1  # High-impact
        else:
            labels[i] = -1  # Low-impact

    # Fallback: if all samples end up in one class, derive labels from relative impact.
    unique_labels = np.unique(labels)
    if unique_labels.size < 2 and len(features) >= 2:
        motion_vals = features[:, 0]
        area_vals = features[:, 2]

        def _normalize(x):
            span = np.ptp(x)
            return np.zeros_like(x) if span == 0 else (x - np.min(x)) / span

        impact_score = _normalize(motion_vals) + _normalize(area_vals)

        n = len(features)
        min_class = int(CONFIG.get('fallback_min_class_samples', 2))
        target_high = int(round(n * CONFIG.get('fallback_high_ratio', 0.30)))

        if n >= 2 * min_class:
            high_count = max(min_class, min(target_high, n - min_class))
        else:
            high_count = max(1, n // 2)

        high_idx = np.argsort(impact_score)[-high_count:]
        labels = -np.ones(n)
        labels[high_idx] = 1

        print("[WARN] Threshold-based auto-labeling produced a single class.")
        print(f"[WARN] Applied fallback labeling using top-{high_count} impact scores as High-impact.")

    return labels

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(X_train, X_test, y_train, y_test):
    """
    Train Logistic Regression model with L2 regularization.
    
    Args:
        X_train, X_test: Training and test feature arrays
        y_train, y_test: Training and test labels
        
    Returns:
        Trained model, StandardScaler
    """
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression with L2 regularization
    model = LogisticRegression(
        penalty='l2',
        C=1.0,  # Inverse of regularization strength
        max_iter=CONFIG['max_iterations'],
        random_state=CONFIG['random_state'],
        solver='lbfgs'
    )
    
    print("\n🤖 Training Logistic Regression model...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 Model Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Low-impact', 'High-impact']))
    print(f"\n  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, scaler

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline."""
    print("=" * 70)
    print("🎬 CINEMATIC IMPACT CLASSIFICATION MODEL")
    print("=" * 70)
    
    # Setup
    setup_directories()
    
    # Load models
    yolo_model = load_yolo_model()
    resnet_model, device = load_resnet_model()
    
    # Get video paths
    if not os.path.isdir(CONFIG['data_dir']):
        print(f"[ERROR] Data directory not found: {CONFIG['data_dir']}")
        return

    video_paths = sorted([
        os.path.join(CONFIG['data_dir'], f)
        for f in os.listdir(CONFIG['data_dir'])
        if any(f.lower().endswith(ext) for ext in CONFIG['video_extensions'])
    ])
    
    if not video_paths:
        print(f"❌ No videos found in {CONFIG['data_dir']}")
        return
    
    print(f"\n📹 Found {len(video_paths)} videos")
    
    # Extract features
    X, video_names = extract_all_features(video_paths, yolo_model, resnet_model, device)
    
    # Auto-label data
    y = auto_label_data(X)
    
    # Count labels
    high_impact = np.sum(y == 1)
    low_impact = np.sum(y == -1)
    print(f"\n📋 Label Distribution:")
    print(f"  High-impact: {high_impact}")
    print(f"  Low-impact: {low_impact}")

    unique_labels = np.unique(y)
    if unique_labels.size < 2:
        print("[ERROR] Need at least 2 classes to train Logistic Regression.")
        print("  Try adjusting thresholds or adding more diverse videos.")
        return

    # Train-test split (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=y
    )
    
    print(f"\n📂 Data Split:")
    print(f"  Train set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # Train model
    model, scaler = train_model(X_train, X_test, y_train, y_test)
    
    # Save model
    with open(CONFIG['model_save_path'], 'wb') as f:
        pickle.dump(model, f)
    print(f"\n💾 Model saved: {CONFIG['model_save_path']}")
    
    # Save scaler
    with open(CONFIG['scaler_save_path'], 'wb') as f:
        pickle.dump(scaler, f)
    print(f"💾 Scaler saved: {CONFIG['scaler_save_path']}")
    
    # Save training summary
    summary_path = os.path.join(CONFIG['output_dir'], 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("CINEMATIC IMPACT CLASSIFICATION MODEL - TRAINING SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total Videos Processed: {len(video_paths)}\n")
        f.write(f"Feature Dimension: {X.shape[1]}\n")
        f.write(f"High-impact Samples: {high_impact}\n")
        f.write(f"Low-impact Samples: {low_impact}\n")
        f.write(f"Train/Test Split: 70/30\n")
        f.write(f"Model Type: Logistic Regression (L2 regularization)\n")
        f.write(f"Test Accuracy: {model.score(scaler.transform(X_test), y_test):.4f}\n")
    print(f"📄 Summary saved: {summary_path}")
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
