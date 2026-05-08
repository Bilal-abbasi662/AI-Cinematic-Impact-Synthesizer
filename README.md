# AI-Cinematic-Impact-Synthesizer

## 🎬 Project Overview
This project is an intelligent system designed to automatically generate high-impact movie trailers by analyzing, selecting, and transforming cinematic scenes. Developed for the **Spring 2026 AI Lab**, it demonstrates a full machine learning pipeline integrating **Computer Vision**, **Audio Analysis**, and **Natural Language Processing**.

## 🧠 Multimodal Pipeline Stages
As per the OEL Lab requirements, the system follows these core steps:

### 1. Feature Extraction & Scene Analysis
We identify "high-impact" scenes based on:
* **Visual Intensity**: Motion, brightness changes, and scene cuts.
* **Object Presence**: Detecting specific elements using **YOLO11**.
* **Audio Features**: Analyzing energy spikes and MFCC (Mel-frequency cepstral coefficients).
* **Emotion & Dynamics**: Facial expressions and camera movement.

### 2. Impact Classification
A classification model is trained to label scenes as **High-Impact (+1)** or **Low-Impact (-1)**.
* **Features**: ResNet/EfficientNet embeddings, YOLO11 features, and MFCC audio energy.
* **Models**: Implementation includes baseline (SVM/Logistic Regression) and temporal models (LSTM/Transformers).
* **Robustness**: Utilizes L2 regularization and cross-validation to prevent overfitting.

### 3. Trailer Generation & "Creepy" Transformation
* **Selection**: 5 clips are chosen based on narrative flow, diversity, and emotional progression.
* **Visual FX**: Detected objects (via YOLO11) are transformed to fit a "Creepy Theme"—including glowing eyes, darkened faces, flicker effects, and fog overlays.

### 4. NLP Captioning
* **Generation**: Automated descriptions using image-captioning models (e.g., BLIP/ViT-GPT2).
* **Sentiment Modulation**: Transforming neutral captions into cinematic, suspenseful overlays (e.g., *"A man walks"* → *"He shouldn't have opened that door..."*).

### 5. Evaluation
The final trailer is passed back through the classifier to generate an **Impact Score Timeline** graph, providing a quantitative measure of the trailer's intensity.

## 🛠️ Technical Stack
* **Detection**: YOLO11
* **Feature Extraction**: PyTorch (ResNet), Librosa (Audio)
* **NLP**: Transformers (HuggingFace), BLIP
* **Data Science**: Scikit-learn, Matplotlib (Evaluation Graphs)

---
*Developed by Bilal Abbasi
