import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import json
from datetime import datetime, timedelta
import requests
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="AI Traffic Control System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79, #2c5282);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3182ce;
    }
    .status-online {
        color: #48bb78;
        font-weight: bold;
    }
    .status-warning {
        color: #ed8936;
        font-weight: bold;
    }
    .emergency-btn {
        background-color: #e53e3e !important;
        color: white !important;
    }
    .camera-feed {
        border-radius: 10px;
        border: 2px solid #3182ce;
        padding: 10px;
        background: #f0f4f8;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'traffic_history' not in st.session_state:
    st.session_state.traffic_history = []
if 'emergency_active' not in st.session_state:
    st.session_state.emergency_active = False
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "AI Adaptive"
if 'esp32_cam_url' not in st.session_state:
    st.session_state.esp32_cam_url = "http://10.85.222.56/"

def load_config():
    """Load configuration from JSON file"""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "intersection_name": "Main St & 1st Ave",
            "detection_confidence": 0.5,
            "esp32_cam_url": "http://10.85.222.56/"
        }

def simulate_vehicle_detection():
    """Simulate vehicle detection"""
    base_counts = {
        'north': np.random.randint(8, 18),
        'south': np.random.randint(6, 15), 
        'east': np.random.randint(10, 20),
        'west': np.random.randint(5, 12)
    }
    
    # Add some realistic variation based on time
    current_hour = datetime.now().hour
    if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:  # Rush hour
        for direction in base_counts:
            base_counts[direction] = int(base_counts[direction] * 1.5)
    
    return base_counts

def update_traffic_history(vehicle_counts):
    """Update traffic history for analytics"""
    timestamp = datetime.now()
    
    # Add current data
    st.session_state.traffic_history.append({
        'timestamp': timestamp,
        'north': vehicle_counts['north'],
        'south': vehicle_counts['south'],
        'east': vehicle_counts['east'],
        'west': vehicle_counts['west'],
        'total': sum(vehicle_counts.values())
    })
    
    # Keep only last 50 records
    if len(st.session_state.traffic_history) > 50:
        st.session_state.traffic_history = st.session_state.traffic_history[-50:]

def display_esp32_cam_feed():
    """Display ESP32-CAM live stream"""
    try:
        st.markdown(f"""
        <div class="camera-feed">
            <h3 style="color: #3182ce; margin-bottom: 1rem;">📹 ESP32-CAM Live Stream</h3>
            <img src="{st.session_state.esp32_cam_url}" width="100%" 
                 style="border-radius: 8px; margin-bottom: 1rem;" 
                 onerror="this.style.display='none'; document.getElementById('cam-error').style.display='block';">
            <div id="cam-error" style="display: none; color: #e53e3e; text-align: center;">
                ⚠️ Camera feed unavailable. Please check ESP32-CAM connection.
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying camera feed: {str(e)}")

def check_camera_connection():
    """Check if ESP32-CAM is accessible"""
    try:
        response = requests.get(st.session_state.esp32_cam_url, timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    config = load_config()
    
    # Update ESP32-CAM URL from config if available
    if 'esp32_cam_url' in config:
        st.session_state.esp32_cam_url = config['esp32_cam_url']
    
    # Header
    st.markdown(f"""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🚦 AI Traffic Control System</h1>
        <h3 style="color: #bee3f8; margin: 0; margin-top: 0.5rem;">
            Real-time Adaptive Management - {config['intersection_name']}
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Controls
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Mode Selection
        st.session_state.current_mode = st.selectbox(
            "Operation Mode", 
            ["AI Adaptive", "Manual Override", "Emergency Mode"]
        )
        
        if st.session_state.current_mode == "AI Adaptive":
            st.success("✅ AI is controlling traffic signals automatically")
            algorithm = st.selectbox(
                "AI Algorithm", 
                ["Adaptive Timing", "Reinforcement Learning", "Genetic Algorithm"],
                help="Select the AI algorithm for traffic optimization"
            )
            
            # AI Parameters
            st.subheader("⚙️ AI Parameters")
            confidence_threshold = st.slider("Detection Confidence", 0.3, 0.9, 0.5)
            min_green_time = st.slider("Min Green Time (s)", 10, 30, 15)
            max_green_time = st.slider("Max Green Time (s)", 40, 90, 60)
            
        elif st.session_state.current_mode == "Manual Override":
            st.warning("⚠️ Manual mode active - AI algorithms disabled")
            north_south_time = st.slider("North-South Green Time (s)", 15, 60, 30)
            east_west_time = st.slider("East-West Green Time (s)", 15, 60, 30)
        
        # ESP32-CAM Settings
        st.subheader("📡 ESP32-CAM Settings")
        st.session_state.esp32_cam_url = st.text_input(
            "ESP32-CAM Stream URL", 
            st.session_state.esp32_cam_url,
            help="Enter the URL for your ESP32-CAM live stream"
        )
        
        # Check camera connection
        cam_status = check_camera_connection()
        st.write(f"Camera Status: {'🟢 Connected' if cam_status else '🔴 Disconnected'}")
        
        # Emergency Controls
        st.subheader("🚨 Emergency Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚑 Emergency Override", key="emergency", help="Activate emergency vehicle priority"):
                st.session_state.emergency_active = True
                st.success("Emergency mode activated!")
        
        with col2:
            if st.button("🔄 Reset Normal", key="reset", help="Return to normal operation"):
                st.session_state.emergency_active = False
                st.info("Normal operation restored")
        
        if st.session_state.emergency_active:
            emergency_direction = st.selectbox(
                "Emergency Vehicle Direction", 
                ["North-South", "East-West"]
            )
            st.error("🚨 EMERGENCY OVERRIDE ACTIVE")
        
        # System Information
        st.subheader("ℹ️ System Info")
        st.info(f"Status: {'Online' if cam_status else 'Offline'}\nLast Update: {datetime.now().strftime('%H:%M:%S')}")

    # Main Dashboard
    # Real-time Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Get current vehicle counts
    vehicle_counts = simulate_vehicle_detection()
    total_vehicles = sum(vehicle_counts.values())
    
    # Update history
    update_traffic_history(vehicle_counts)
    
    with col1:
        status = "🟢 Online" if not st.session_state.emergency_active else "🟡 Emergency"
        delta_status = "All systems operational" if not st.session_state.emergency_active else "Emergency mode"
        st.metric("System Status", status, delta=delta_status)
    
    with col2:
        delta_vehicles = np.random.randint(-5, 8)
        delta_sign = "+" if delta_vehicles >= 0 else ""
        st.metric("Total Vehicles", total_vehicles, delta=f"{delta_sign}{delta_vehicles}")
    
    with col3:
        avg_wait = round(2.3 + np.random.uniform(-0.5, 0.5), 1)
        wait_delta = round(np.random.uniform(-0.8, 0.3), 1)
        delta_sign = "+" if wait_delta >= 0 else ""
        st.metric("Avg Wait Time", f"{avg_wait} min", delta=f"{delta_sign}{wait_delta} min")
    
    with col4:
        efficiency = round(94.2 + np.random.uniform(-2, 3), 1)
        eff_delta = round(np.random.uniform(-1, 4), 1)
        delta_sign = "+" if eff_delta >= 0 else ""
        st.metric("AI Efficiency", f"{efficiency}%", delta=f"{delta_sign}{eff_delta}%")

    # Live Traffic Monitoring with ESP32-CAM
    st.subheader(f"📹 Live Traffic Monitoring - {config['intersection_name']}")
    
    # Create layout with camera feed and traffic data
    cam_col, data_col = st.columns([2, 1])
    
    with cam_col:
        # ESP32-CAM Live Stream
        display_esp32_cam_feed()
        
        # Additional camera information
        st.info(f"📡 Connected to: {st.session_state.esp32_cam_url}")
        
    with data_col:
        st.subheader("🚦 Signal Status")
        
        # Calculate adaptive timing
        if not st.session_state.emergency_active and st.session_state.current_mode == "AI Adaptive":
            # AI-calculated timing
            total_ns = vehicle_counts['north'] + vehicle_counts['south']
            total_ew = vehicle_counts['east'] + vehicle_counts['west']
            
            if total_ns > total_ew:
                ns_signal = "🟢"
                ew_signal = "🔴"
                ns_timer = min(60, max(15, total_ns * 2))
                ew_timer = 60 - ns_timer + 15
            else:
                ns_signal = "🔴"
                ew_signal = "🟢" 
                ew_timer = min(60, max(15, total_ew * 2))
                ns_timer = 60 - ew_timer + 15
        else:
            # Default timing
            ns_signal = "🟢"
            ew_signal = "🔴"
            ns_timer = 30
            ew_timer = 30
        
        st.write(f"{ns_signal} **North-South**: {ns_timer}s")
        st.write(f"{ew_signal} **East-West**: {ew_timer}s")
        
        st.subheader("🤖 AI Status")
        st.success("✅ Adaptive Control: Active") 
        st.info(f"🧠 Confidence: {confidence_threshold*100:.1f}%")
        st.info(f"🔄 Processing: {np.random.randint(28, 35)} FPS")
        
        # Vehicle counts by direction
        st.subheader("🚗 Vehicle Counts")
        for direction, count in vehicle_counts.items():
            st.write(f"**{direction.title()}**: {count} vehicles")

    # Vehicle count display in a separate section
    st.subheader("📊 Lane-specific Vehicle Counts")
    subcol1, subcol2, subcol3, subcol4 = st.columns(4)
    
    directions = ['north', 'south', 'east', 'west']
    direction_icons = ['⬆️', '⬇️', '➡️', '⬅️']
    direction_names = ['North', 'South', 'East', 'West']
    
    for i, (direction, icon, name) in enumerate(zip(directions, direction_icons, direction_names)):
        with [subcol1, subcol2, subcol3, subcol4][i]:
            count = vehicle_counts[direction]
            
            # Color coding based on density
            if count < 8:
                density_color = "🟢"
                density = "Low"
            elif count < 15:
                density_color = "🟡" 
                density = "Medium"
            else:
                density_color = "🔴"
                density = "High"
            
            st.metric(
                f"{density_color} {icon} {name} Lane", 
                count,
                delta=f"{density} density"
            )

    # Analytics Dashboard
    st.subheader("📈 Real-time Analytics & Performance")
    
    # Charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        if len(st.session_state.traffic_history) > 1:
            # Traffic trends
            df_history = pd.DataFrame(st.session_state.traffic_history)
            
            fig_line = px.line(
                df_history, 
                x='timestamp', 
                y=['north', 'south', 'east', 'west'],
                title="🚗 Vehicle Count Trends (Real-time)",
                labels={'value': 'Vehicle Count', 'timestamp': 'Time'},
                color_discrete_map={
                    'north': '#3182ce',
                    'south': '#e53e3e', 
                    'east': '#38a169',
                    'west': '#d69e2e'
                }
            )
            fig_line.update_layout(height=350)
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Collecting traffic data... Please wait.")
    
    with chart_col2:
        # Current distribution
        fig_pie = px.pie(
            values=list(vehicle_counts.values()),
            names=[f"{name} Lane" for name in ['North', 'South', 'East', 'West']],
            title="🚦 Current Traffic Distribution",
            color_discrete_map={
                'North Lane': '#3182ce',
                'South Lane': '#e53e3e',
                'East Lane': '#38a169', 
                'West Lane': '#d69e2e'
            }
        )
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Auto-refresh
    time.sleep(5)
    st.rerun()

if __name__ == "__main__":
    main()
