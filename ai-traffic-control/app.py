import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta

st.set_page_config(
    page_title="AI Traffic Control System",
    page_icon="🚦",
    layout="wide"
)

# Initialize session state
if 'emergency_mode' not in st.session_state:
    st.session_state.emergency_mode = False
if 'mode' not in st.session_state:
    st.session_state.mode = "AI Adaptive"
if 'emergency_dir' not in st.session_state:
    st.session_state.emergency_dir = "North-South"
if 'pedestrian_priority' not in st.session_state:
    st.session_state.pedestrian_priority = False
if 'lane_reversed' not in st.session_state:
    st.session_state.lane_reversed = False
if 'esp32_ip' not in st.session_state:
    # 👉 Replace with your ESP32-CAM IP address after uploading ESP32 sketch
    st.session_state.esp32_ip = "http://192.168.1.50"

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')

def main():
    # Header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1f4e79, #2c5282); padding: 2rem; border-radius: 10px; margin-bottom: 2rem; text-align: center;">
        <h1 style="color: white; margin: 0;">🚦 AI Traffic Control System</h1>
        <h3 style="color: #bee3f8; margin: 0; margin-top: 0.5rem;">Real-time Adaptive Management - Main St & 1st Ave</h3>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        st.session_state.mode = st.selectbox(
            "Operation Mode", 
            ["AI Adaptive", "Manual Override", "Emergency Mode", "Predictive Mode"]
        )

        # Emergency Controls
        st.subheader("🚨 Emergency Controls")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚑 Activate Emergency", type="primary"):
                st.session_state.emergency_mode = True
        with col2:
            if st.button("🔄 Deactivate"):
                st.session_state.emergency_mode = False

        if st.session_state.emergency_mode:
            st.session_state.emergency_dir = st.selectbox("Emergency Direction", ["North-South", "East-West"])
            st.error("🚨 EMERGENCY OVERRIDE ACTIVE")

        # Futuristic Features
        st.subheader("🚀 Futuristic Features")
        st.session_state.lane_reversed = st.checkbox("Dynamic Lane Management", value=st.session_state.lane_reversed)
        st.session_state.pedestrian_priority = st.checkbox("Pedestrian & Cyclist Priority", value=st.session_state.pedestrian_priority)

        # ESP32 Camera IP input
        st.subheader("📡 ESP32-CAM Settings")
        st.session_state.esp32_ip = st.text_input("ESP32-CAM Stream URL", st.session_state.esp32_ip)

    # Metrics
    vehicle_counts = {
        'north': np.random.randint(8, 18),
        'south': np.random.randint(6, 15), 
        'east': np.random.randint(10, 20),
        'west': np.random.randint(5, 12)
    }
    total_vehicles = sum(vehicle_counts.values())

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("System Status", "🟢 Online" if not st.session_state.emergency_mode else "🟡 Emergency")
    with col2:
        st.metric("Total Vehicles", total_vehicles)
    with col3:
        st.metric("Avg Wait Time", f"{round(2.3 + np.random.uniform(-0.5, 0.5), 1)} min")
    with col4:
        st.metric("AI Efficiency", f"{round(94.2 + np.random.uniform(-2, 3), 1)}%")

    # 🚦 Live Traffic Monitoring + ESP32 Feed
    st.subheader("📹 Live Traffic Monitoring - Main St & 1st Ave")
    col_main, col_signals = st.columns([3, 1])

    with col_main:
        st.info("🎥 Live ESP32-CAM Feed")
        if st.session_state.esp32_ip:
            stream_url = st.session_state.esp32_ip + ":81/stream"
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center;">
                    <img src="{stream_url}" width="100%" style="border-radius:10px;"/>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("⚠️ Please enter your ESP32-CAM IP in the sidebar.")

    with col_signals:
        st.subheader("🚦 Signal Status")
        st.write("🔄 Adaptive signals displayed here...")

    # Auto-refresh every 5 seconds
    time.sleep(5)
    st.rerun()

if __name__ == "__main__":
    main()
