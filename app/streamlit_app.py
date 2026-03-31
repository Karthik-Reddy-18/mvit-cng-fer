"""
streamlit_app.py — Interactive web application for emotion recognition
Run with: streamlit run app/streamlit_app.py
"""
 
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import yaml
import sys
import os
from pathlib import Path
 
# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import MViTCnG
from preprocess import get_test_transforms
from face_detector import FaceDetector
 
# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Recognizer — MViT-CnG",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ─── Custom CSS Styling ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: #1F4E79;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    .emotion-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-card {
        background: #f0f4ff;
        border-left: 4px solid #1F4E79;
        padding: 10px 15px;
        border-radius: 0 10px 10px 0;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)
 
# ─── Emotion Configuration ────────────────────────────────────────────────────
EMOTION_EMOJIS = {
    "angry":    "😡",
    "disgust":  "🤢",
    "fear":     "😨",
    "happy":    "😊",
    "sad":      "😢",
    "surprise": "😲",
    "neutral":  "😐"
}
 
EMOTION_COLORS = {
    "angry":    "#FF4444",
    "disgust":  "#8BC34A",
    "fear":     "#9C27B0",
    "happy":    "#FFD700",
    "sad":      "#2196F3",
    "surprise": "#FF9800",
    "neutral":  "#9E9E9E"
}
 
# ─── Model Loading (cached for performance) ───────────────────────────────────
@st.cache_resource
def load_model():
    """Load model once and cache it — prevents reloading on every interaction."""
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)
    
    num_classes = cfg['data']['num_classes']
    emotions = cfg['emotions']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MViTCnG(
        num_classes=num_classes,
        image_size=cfg['data']['image_size'],
        embed_dim=256,
        num_heads=cfg['model']['num_heads'],
        num_layers=6,
        dropout=0.0,
        contrastive_dim=cfg['model']['contrastive_dim']
    ).to(device)
    
    checkpoint_path = "models/mvit_cng_fer2013.pth"
    
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        st.sidebar.success(f"✅ Model loaded!")
    else:
        st.sidebar.error(f"❌ Model file not found: {checkpoint_path}")
        st.sidebar.info("Please train the model first using: python src/train.py")
    
    return model, emotions, device
 
@st.cache_resource
def load_detector():
    return FaceDetector()
 
# ─── Prediction Function ──────────────────────────────────────────────────────
def predict_emotion(model, image_rgb, emotions, device):
    """
    Given an RGB image, detect face and predict emotion.
    Returns: (emotion_label, confidence%, annotated_image)
    """
    detector = load_detector()
    
    # Convert to BGR for OpenCV
    image_bgr = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)
    
    # Detect faces
    faces, boxes = detector.detect_faces(image_bgr)
    
    if len(faces) == 0:
        # No face detected — try using entire image
        st.warning("⚠️ No face detected. Analyzing entire image.")
        faces = [image_bgr]
        boxes = [(0, 0, image_bgr.shape[1], image_bgr.shape[0])]
    
    results = []
    transform = get_test_transforms()
    
    for face_bgr in faces:
        # Convert to PIL RGB
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        
        # Preprocess
        tensor = transform(face_pil).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            logits, _ = model(tensor)
            probs = torch.softmax(logits, dim=-1)[0]
        
        predicted_idx = torch.argmax(probs).item()
        confidence = probs[predicted_idx].item() * 100
        emotion = emotions[predicted_idx]
        
        results.append({
            'emotion': emotion,
            'confidence': confidence,
            'all_probs': {emotions[i]: probs[i].item() * 100 for i in range(len(emotions))}
        })
    
    # Draw on original image
    annotated = image_bgr.copy()
    for (x, y, w, h), res in zip(boxes, results):
        color_hex = EMOTION_COLORS.get(res['emotion'], '#00FF00')
        r, g, b = int(color_hex[1:3],16), int(color_hex[3:5],16), int(color_hex[5:7],16)
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (b, g, r), 3)
        label = f"{EMOTION_EMOJIS.get(res['emotion'], '')} {res['emotion']} ({res['confidence']:.1f}%)"
        cv2.rectangle(annotated, (x, y-35), (x+w, y), (b, g, r), -1)
        cv2.putText(annotated, label, (x+5, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return results, annotated_rgb
 
# ─── Main App UI ──────────────────────────────────────────────────────────────
def main():
    # Title
    st.markdown('<div class="main-title">🎭 Facial Emotion Recognizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Powered by Multi-Scale Vision Transformer with Contrastive Learning</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    model, emotions, device = load_model()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Device:** {'🟢 GPU' if device.type == 'cuda' else '🔵 CPU'}")
    st.sidebar.markdown(f"**Classes:** {', '.join(emotions)}")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["📁 Upload Image", "📸 Use Webcam"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if input_method == "📁 Upload Image":
        uploaded_file = st.file_uploader(
            "Upload a face image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload any image containing a human face"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Input Image")
                st.image(image, use_column_width=True)
            
            with st.spinner("🔍 Detecting face and analyzing emotion..."):
                results, annotated = predict_emotion(model, image, emotions, device)
            
            with col2:
                st.subheader("🎯 Detection Result")
                st.image(annotated, use_column_width=True)
            
            if results:
                st.markdown("---")
                st.subheader("📊 Emotion Analysis")
                
                for res in results:
                    st.markdown(
                        f'<div class="emotion-card">'
                        f'{EMOTION_EMOJIS.get(res["emotion"], "🎭")} '
                        f'{res["emotion"].upper()} — {res["confidence"]:.1f}%'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                # Probability bars
                st.subheader("📈 All Emotion Probabilities")
                probs = results[0]['all_probs']
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                
                for emotion, prob in sorted_probs:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.progress(int(prob), text=f"{EMOTION_EMOJIS.get(emotion,'')} {emotion}")
                    with col2:
                        st.write(f"**{prob:.1f}%**")
    
    else:  # Webcam mode
        st.info("📸 Click 'Start Camera' to use your webcam")
        img_file_buffer = st.camera_input("Take a photo")
        
        if img_file_buffer is not None:
            image = Image.open(img_file_buffer)
            
            with st.spinner("🔍 Analyzing..."):
                results, annotated = predict_emotion(model, image, emotions, device)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.image(image, use_column_width=True)
            with col2:
                st.subheader("Detection")
                st.image(annotated, use_column_width=True)
            
            if results:
                emotion = results[0]['emotion']
                conf = results[0]['confidence']
                st.success(f"{EMOTION_EMOJIS.get(emotion, '🎭')} Detected: **{emotion.upper()}** ({conf:.1f}% confidence)")
 
if __name__ == "__main__":
    main()
 
