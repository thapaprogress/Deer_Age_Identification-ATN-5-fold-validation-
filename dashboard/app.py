import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
from pathlib import Path

import os
import sys

# Ensure project root is in path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

# Custom Modules
from utils.inference import InferenceEngine
from utils.gradcam import GradCAM, overlay_heatmap
from dashboard.tracker import CentroidTracker

# Page Config
st.set_page_config(
    page_title="ATN Deer Age Recognition System",
    page_icon="🦌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Global Theme */
    .stApp {
        background-color: #f8f9fa !important;
        color: #31333F !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2c3e50 !important;
        font-weight: 700;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9ecef;
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stMetricLabel"] {
        color: #6c757d;
        font-size: 0.9rem;
    }
    div[data-testid="stMetricValue"] {
        color: #2c3e50;
        font-weight: 700;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e9ecef;
    }
    
    /* Custom Containers/Cards for other content */
    .css-1r6slb0 {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Upload header */
    .css-1qg05tj {
        color: #2c3e50;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_inference_engine():
    """Load single model once and cache it"""
    checkpoint_dir = ROOT_DIR / "checkpoints"
    model_path = checkpoint_dir / "best_model.pth"
    
    if not model_path.exists():
        st.warning("⚠️ 'best_model.pth' not found. Using random weights for demo.")
        return InferenceEngine()
    
    return InferenceEngine(model_path=str(model_path))

def main():
    st.title("🦌 ATN Deer Age Recognition System")
    st.markdown("### Augmented Triplet Network for Wildlife Monitoring")
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", ["Dashboard", "Image Analysis", "Video Analysis", "Training Monitor"])
    
    # Load Model
    with st.spinner("Loading ATN Model..."):
        engine = load_inference_engine()
        
    if app_mode == "Dashboard":
        render_dashboard(engine)
    elif app_mode == "Image Analysis":
        render_image_analysis(engine)
    elif app_mode == "Video Analysis":
        render_video_analysis(engine)
    elif app_mode == "Training Monitor":
        render_training_monitor()

def render_dashboard(engine):
    st.header("📊 System Overview")
    
    c1, c2, c3, c4 = st.columns(4)
    
    num_ref_images = len(engine.reference_labels) if engine.knn else 0
    num_classes = len(np.unique(engine.reference_labels)) if engine.knn else 0
    
    c1.metric("Total Reference Images", num_ref_images)
    c2.metric("Age Classes", num_classes, "2-8 Years")
    c3.metric("Model Architecture", "ResNet-18 + ATN")
    c4.metric("Inference Device", engine.device.upper())
    
    if engine.knn:
        st.subheader("Reference Data Distribution")
        df_dist = pd.DataFrame({'Age': engine.reference_labels})
        fig = px.histogram(df_dist, x='Age', nbins=20, title="Training Data Profile",
                           color_discrete_sequence=['#4CAF50'])
        st.plotly_chart(fig, use_container_width=True)

def render_image_analysis(engine):
    st.header("📸 Single Image Analysis")
    
    uploaded_file = st.file_uploader("Upload Deer Image", type=['jpg', 'jpeg', 'png', 'heic'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        if 'image_result' not in st.session_state or st.session_state.get('last_image_name') != uploaded_file.name:
            st.session_state.image_result = None
            st.session_state.last_image_name = uploaded_file.name
            st.session_state.heatmap = None
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            show_gradcam = st.toggle("Show Attention Map (Grad-CAM)", value=True)
            
        with c2:
            st.subheader("Analysis Control")
            if st.button("🚀 Run Analysis", type="primary"):
                with st.spinner("🔍 Estimating age..."):
                    # Basic prediction (No TTA)
                    age, conf, emb = engine.predict(image, use_tta=False)
                    st.session_state.image_result = {'age': age, 'conf': conf}
                    
                    if show_gradcam:
                        try:
                            # Target layer for Grad-CAM
                            target_layer = engine.model.backbone.resnet[7][-1].conv2
                            gradcam = GradCAM(engine.model, target_layer)
                            img_tensor = engine.transform(image).unsqueeze(0).to(engine.device)
                            heatmap = gradcam(img_tensor)
                            st.session_state.heatmap = heatmap
                        except Exception as e:
                            st.error(f"Attention Map Failed: {e}")
            
            if st.session_state.image_result:
                res = st.session_state.image_result
                st.success(f"### Predicted Age: {res['age']} Years")
                st.caption(f"Confidence Score: {res['conf']:.2%}")
                
                if show_gradcam and st.session_state.heatmap is not None:
                    st.divider()
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        image.save(tmp.name)
                        overlay = overlay_heatmap(tmp.name, st.session_state.heatmap)
                    os.unlink(tmp.name)
                    st.image(overlay, caption="Attention Map Overlay", channels="BGR", use_container_width=True)
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = res['age'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Estimated Age"},
                    gauge = {
                        'axis': {'range': [1, 9], 'tickwidth': 1},
                        'bar': {'color': "#4CAF50"},
                        'steps': [
                            {'range': [1, 4], 'color': "#f0f0f0"},
                            {'range': [4, 7], 'color': "#e0e0e0"},
                            {'range': [7, 9], 'color': "#d0d0d0"}]
                    }
                    ))
                st.plotly_chart(fig, use_container_width=True)

def merge_boxes(rects, threshold=0.1):
    if not rects: return []
    rects = sorted(rects, key=lambda x: x[0])
    merged = []
    while rects:
        curr = list(rects.pop(0))
        i = 0
        while i < len(rects):
            other = rects[i]
            x1 = max(curr[0], other[0]); y1 = max(curr[1], other[1])
            x2 = min(curr[2], other[2]); y2 = min(curr[3], other[3])
            if x1 < x2 and y1 < y2:
                curr[0] = min(curr[0], other[0]); curr[1] = min(curr[1], other[1])
                curr[2] = max(curr[2], other[2]); curr[3] = max(curr[3], other[3])
                rects.pop(i); i = 0
            else: i += 1
        merged.append(tuple(curr))
    return merged

def get_id_color(objectID):
    colors = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,128,0),(0,128,255)]
    return colors[objectID % len(colors)]

def render_video_analysis(engine):
    st.header("🎥 Video Analysis")
    uploaded_video = st.file_uploader("Upload Deer Video", type=['mp4', 'avi', 'mov'])
    if uploaded_video is not None:
        if 'video_path' not in st.session_state or st.session_state.get('last_uploaded_name') != uploaded_video.name:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                tfile.write(uploaded_video.read())
                st.session_state.video_path = tfile.name
                st.session_state.last_uploaded_name = uploaded_video.name
        temp_path = st.session_state.video_path
        if 'is_tracking' not in st.session_state: st.session_state.is_tracking = False
        if st.button("▶️ Start Analysis"): st.session_state.is_tracking = True
        stframe = st.empty()
        if st.session_state.is_tracking:
            cap = cv2.VideoCapture(temp_path)
            tracker = CentroidTracker(maxDistance=120, maxDisappeared=50)
            track_history = {}
            fgbg = cv2.createBackgroundSubtractorMOG2()
            frame_count = 0
            try:
                while cap.isOpened() and st.session_state.is_tracking:
                    ret, frame = cap.read()
                    if not ret: break
                    frame_count += 1
                    if frame_count % 2 != 0: continue
                    frame = cv2.resize(frame, (640, 480))
                    fgmask = fgbg.apply(cv2.GaussianBlur(frame, (5, 5), 0))
                    contours, _ = cv2.findContours(cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, np.ones((5,5))), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    raw_rects = [(x, y, x+w, y+h) for cnt in contours if cv2.contourArea(cnt) > 2000 for x, y, w, h in [cv2.boundingRect(cnt)]]
                    rects = merge_boxes(raw_rects)
                    objects = tracker.update(rects)
                    for (objectID, centroid) in objects.items():
                        color = get_id_color(objectID)
                        cx, cy = centroid
                        if objectID not in track_history: track_history[objectID] = {'age': 0}
                        if track_history[objectID]['age'] == 0:
                            try:
                                crop = frame[max(0,cy-120):min(480,cy+120), max(0,cx-120):min(640,cx+120)]
                                age, _, _ = engine.predict(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), use_tta=False)
                                track_history[objectID]['age'] = age
                            except: pass
                        for (x1, y1, x2, y2) in rects:
                            if x1 <= cx <= x2 and y1 <= cy <= y2:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                break
                        cv2.putText(frame, f"ID {objectID}: {track_history[objectID]['age']}Y", (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    stframe.image(frame, channels="BGR")
                    time.sleep(0.01)
            finally: cap.release()

def render_training_monitor():
    st.header("📈 Training Monitor")
    log_dir = ROOT_DIR / "logs"
    if log_dir.exists(): st.success(f"Logs: {log_dir}")
    else: st.error("No logs found.")

if __name__ == "__main__":
    main()
