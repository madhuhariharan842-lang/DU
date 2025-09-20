import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
from ultralytics import YOLO
import time

# Configs and YOLO Model (ambulance priority)
YOLO_MODEL = 'yolov8n.pt'   # Ensure you have this downloaded
AMBULANCE_LABELS = {'ambulance', 'Ambulance'}
FOUR_LANES = ['north', 'south', 'east', 'west']
CAM_STREAM_URL = "http://10.85.222.56:81/stream"  # ESP32-CAM stream
MODEL = YOLO(YOLO_MODEL)

st.set_page_config(page_title="AI Adaptive Traffic Signal", layout="wide")
st.title("🚦 Adaptive AI Traffic Signal with Emergency Mode")

# Sidebar sensors/config
st.sidebar.header("Sensors & Environment")
humidity = st.sidebar.slider("Humidity level (%)", 0, 100, 35)
conf_threshold = st.sidebar.slider("YOLO Confidence Threshold", 0.1, 1.0, 0.5, step=0.05)
source_mode = st.sidebar.selectbox("Video Source", ["Webcam", "ESP32-CAM", "Video Upload"])
emergency_override = st.sidebar.checkbox("Manual Emergency Mode (Ambulance)", False)

# Prepare Stream Source
def get_video_capture():
    if source_mode == "Webcam":
        return cv2.VideoCapture(0)
    elif source_mode == "ESP32-CAM":
        return cv2.VideoCapture(CAM_STREAM_URL)
    elif source_mode == "Video Upload":
        up = st.sidebar.file_uploader("Upload MP4", type=['mp4', 'avi', 'mov'])
        if up:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(up.read())
            return cv2.VideoCapture(tfile.name)
    return None

# State Management
if 'lane_counts' not in st.session_state:
    st.session_state.lane_counts = {lane: [] for lane in FOUR_LANES}
    st.session_state.time_ticks = []
    st.session_state.last_signal = "north_south"
    st.session_state.signal_timer = time.time()
    st.session_state.ambulance_detected = False

# Main App
col_left, col_mid, col_right = st.columns([1,2,1])
with col_left:
    st.subheader("Lane Vehicle Counts")
    status_zone = st.empty()

with col_mid:
    st.subheader("Live Traffic Detection & Signal State")

    FRAME_WINDOW = st.empty()
    INFO_BOX = st.empty()
    signal_display = st.empty()
    density_plot = st.empty()

with col_right:
    st.subheader("Detection & Emergency Panel")
    amb_box = st.empty()
    humidity_box = st.empty()

def yolo_predict(frame):
    results = MODEL(frame, conf=conf_threshold)
    pred_labels = []
    pred_boxes = []
    for r in results:
        for c,box,conf in zip(r.boxes.cls, r.boxes.xyxy, r.boxes.conf):
            label = MODEL.model.names[int(c)]
            if conf >= conf_threshold:
                pred_labels.append(label)
                pred_boxes.append(box.cpu().numpy())
    return pred_labels, pred_boxes

def classify_ambulance(labels):
    for x in labels:
        if any(lbl.lower() in x.lower() for lbl in AMBULANCE_LABELS):
            return True
    return False

def count_per_lane(boxes, frame_shape):
    # Divide the frame into four quadrants for N/S/E/W
    h,w = frame_shape[:2]
    counts = {'north':0,'south':0,'east':0,'west':0}
    for x1,y1,x2,y2 in boxes:
        cx,cy = int((x1+x2)//2), int((y1+y2)//2)
        if cy<h//2 and cx<w//2:
            counts['north']+=1
        elif cy>h//2 and cx<w//2:
            counts['west']+=1
        elif cy<h//2 and cx>w//2:
            counts['east']+=1
        else:
            counts['south']+=1
    return counts

def auto_signal_decision(lane_counts, ambulance, humidity):
    # Priority: Ambulance > Heaviest Queue > Direction cycling
    if ambulance and st.session_state.last_signal != ambulance:
        return {'signal': ambulance, 'duration': 45, 'reason':'EMERGENCY'}
    # If no ambulance, pick lane with max traffic; modify with humidity
    max_lane = max(lane_counts, key=lane_counts.get)
    min_green = 15
    max_green = 45
    base_duration = int(min_green + (max_green-min_green)*(lane_counts[max_lane]/max(1,sum(lane_counts.values()))))
    adj_duration = int(base_duration * (1.2 if humidity>80 else 1))
    return {'signal': max_lane, 'duration': min(adj_duration,max_green), 'reason':'DENSITY_OPTIMIZED'}

# Real-Time loop logic
cap = get_video_capture()
run = st.button("Start Signal AI")
stop = st.button("Stop Signal")
last_update = time.time()
if cap and run:
    stop_signal = False
    while not stop_signal:
        ret, frame = cap.read()
        if not ret:
            st.warning("No frame received. (Check camera/video source)")
            break

        # YOLO detection
        labels, boxes = yolo_predict(frame)
        ambulance_detected = classify_ambulance(labels) or emergency_override
        counts = count_per_lane(boxes, frame.shape)
        st.session_state.time_ticks.append(time.time())

        # Store lane counts
        for lane in FOUR_LANES:
            st.session_state.lane_counts[lane].append(counts[lane])

        # Emergency override
        emergency_lane = None
        if ambulance_detected:
            # Assign signal to lane where ambulance most likely detected
            lane_scores = {lane:st.session_state.lane_counts[lane][-1] for lane in FOUR_LANES}
            emergency_lane = max(lane_scores, key=lane_scores.get)
        # Signal optimization
        signal_decision = auto_signal_decision(counts, emergency_lane, humidity)
        st.session_state.last_signal = signal_decision['signal']

        # Visualization!
        frame_disp = frame.copy()
        for b in boxes:
            x1,y1,x2,y2 = map(int,b)
            color=(0,0,255) if ambulance_detected else (0,255,0)
            cv2.rectangle(frame_disp,(x1,y1),(x2,y2),color,3)
        FRAME_WINDOW.image(cv2.cvtColor(frame_disp,cv2.COLOR_BGR2RGB),channels="RGB",caption="YOLO Detection")

        # Current state info
        INFO_BOX.info(f"Current Densities: {counts} | Emergency: {ambulance_detected}")
        signal_display.warning(f"🔁 Signal Green: {signal_decision['signal'].upper()} (Reason: {signal_decision['reason']}) | Duration: {signal_decision['duration']} sec")
        amb_box.success("🚨 Ambulance Detected! EMERGENCY MODE" if ambulance_detected else "No Emergency")
        humidity_box.info(f"🌡️ Humidity: {humidity}%")
        status_zone.text(f"Signal: {signal_decision['signal'].upper()} - Duration: {signal_decision['duration']}s")

        # Graph
        if len(st.session_state.lane_counts['north']) > 2:
            t = st.session_state.time_ticks[-100:]
            for lane in FOUR_LANES:
                if len(st.session_state.lane_counts[lane])>100:
                    st.session_state.lane_counts[lane] = st.session_state.lane_counts[lane][-100:]
            fig = go.Figure()
            for lane in FOUR_LANES:
                fig.add_trace(go.Scatter(x=list(range(len(st.session_state.lane_counts[lane]))),
                    y=st.session_state.lane_counts[lane], mode='lines+markers', name=lane))
            density_plot.plotly_chart(fig, use_container_width=True)

        # Timer for signal change
        now = time.time()
        if now - last_update > signal_decision['duration']:
            last_update = now
            st.session_state.last_signal = signal_decision['signal']

        # Slow down loop for Streamlit UI, but not for actual hardware  
        time.sleep(1)

    cap.release()
    st.success("Signal stopped or video ended.")

st.write("System Ready! Configure inputs at left and press 'Start Signal AI' to begin adaptive operation.")
