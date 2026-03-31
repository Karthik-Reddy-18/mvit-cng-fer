"""
streamlit_app.py — Facial Emotion Recognition Web App
Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import yaml
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import MViTCnG
from torchvision import transforms

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Recognizer",
    page_icon="😊",
    layout="wide"
)

# ─── Constants ────────────────────────────────────────────────────────────────
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

EMOJI = {
    "angry":    "😡",
    "disgust":  "🤢",
    "fear":     "😨",
    "happy":    "😊",
    "sad":      "😢",
    "surprise": "😲",
    "neutral":  "😐"
}

# ─── Transforms ───────────────────────────────────────────────────────────────
def get_transform():
    return transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MViTCnG(
        num_classes=7,
        image_size=48,
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        dropout=0.0,
        contrastive_dim=128
    ).to(device)

    model_path = Path("models/mvit_cng_fer2013.pth")

    if not model_path.exists():
        return None, device

    checkpoint = torch.load(str(model_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device

# ─── Face detection ───────────────────────────────────────────────────────────
def detect_face(image_bgr):
    """Returns cropped face or original image if no face found."""
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces  = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return image_bgr, None

    # Use the largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = image_bgr[y:y+h, x:x+w]
    return face, (x, y, w, h)

# ─── Predict ──────────────────────────────────────────────────────────────────
def predict(model, device, image_pil):
    """Takes PIL image, returns (emotion, confidence, all_probs dict)."""
    transform = get_transform()

    # Convert to BGR for face detection
    image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    face_bgr, box = detect_face(image_bgr)

    # Convert face back to RGB PIL for transform
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)

    tensor = transform(face_pil).unsqueeze(0).to(device)  # [1, 3, 48, 48]

    with torch.no_grad():
        logits, _ = model(tensor)
        probs     = F.softmax(logits, dim=-1)[0]

    pred_idx   = torch.argmax(probs).item()
    emotion    = EMOTIONS[pred_idx]
    confidence = probs[pred_idx].item() * 100

    all_probs = {EMOTIONS[i]: round(probs[i].item() * 100, 2)
                 for i in range(len(EMOTIONS))}

    return emotion, confidence, all_probs, box

# ─── Draw box on image ────────────────────────────────────────────────────────
def draw_box(image_pil, box, emotion, confidence):
    """Draw bounding box and label on image."""
    img = np.array(image_pil)
    if box is not None:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 200, 0), 2)
        label = f"{emotion} {confidence:.1f}%"
        cv2.rectangle(img, (x, y-30), (x+w, y), (0, 200, 0), -1)
        cv2.putText(img, label, (x+4, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return Image.fromarray(img)

# ─── Main UI ──────────────────────────────────────────────────────────────────
def main():
    st.title("🎭 Facial Emotion Recognizer")
    st.caption("Powered by Multi-Scale Vision Transformer with Contrastive Learning")

    # Load model
    model, device = load_model()

    if model is None:
        st.error("❌ Model file not found: models/mvit_cng_fer2013.pth")
        st.info("""
        **To fix this:**
        1. Train the model locally:  `python src/train.py`
        2. This creates `models/mvit_cng_fer2013.pth`
        3. Push it to GitHub (use Git LFS if file > 100MB)
        4. Reboot this app
        """)
        st.stop()

    st.sidebar.success("✅ Model loaded!")
    st.sidebar.markdown(f"**Device:** {'🟢 GPU' if device.type == 'cuda' else '🔵 CPU'}")
    st.sidebar.markdown("**Emotions:** " + " · ".join(
        [f"{EMOJI[e]} {e}" for e in EMOTIONS]
    ))

    # Input method
    mode = st.radio("Input method:", ["📁 Upload Image", "📸 Webcam"], horizontal=True)
    st.markdown("---")

    image = None

    if mode == "📁 Upload Image":
        uploaded = st.file_uploader(
            "Upload a face image",
            type=["jpg", "jpeg", "png", "bmp", "webp"]
        )
        if uploaded:
            image = Image.open(uploaded).convert("RGB")

    else:
        captured = st.camera_input("Take a photo")
        if captured:
            image = Image.open(captured).convert("RGB")

    # Run prediction
    if image is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input")
            st.image(image, use_column_width=True)

        with st.spinner("Analysing..."):
            try:
                emotion, confidence, all_probs, box = predict(model, device, image)
                annotated = draw_box(image, box, emotion, confidence)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        with col2:
            st.subheader("Result")
            st.image(annotated, use_column_width=True)

        st.markdown("---")
        st.subheader("Emotion Detected")

        # Big result card
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,#1F4E79,#2E75B6);
                        color:white;padding:20px;border-radius:12px;
                        text-align:center;font-size:2rem;font-weight:bold;">
                {EMOJI.get(emotion, '🎭')} {emotion.upper()} &nbsp;—&nbsp; {confidence:.1f}%
            </div>
            """,
            unsafe_allow_html=True
        )

        # Show only top emotion — nothing else
        top_emotion = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[0]
        emo, prob   = top_emotion

        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,#1F4E79,#2E75B6);
                        color:white; padding:30px; border-radius:15px;
                        text-align:center;">
                <div style="font-size:4rem;">{EMOJI.get(emo, '🎭')}</div>
                <div style="font-size:2rem; font-weight:bold; margin-top:10px;">
                    {emo.upper()}
                </div>
                <div style="font-size:1.2rem; margin-top:8px; opacity:0.85;">
                    Confidence: {prob:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
