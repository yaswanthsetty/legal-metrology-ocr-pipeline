"""
Legal Metrology OCR Compliance Pipeline - Streamlit Web Interface

A professional web interface for automated Legal Metrology compliance validation
using advanced Computer Vision and AI.

Author: Legal Metrology OCR Pipeline Team
License: MIT
"""

import streamlit as st
import os
import sys
import time
import json
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import base64

# Add project root to path for imports
project_root = Path(__file__).parent.parent  # Go up one level from web folder 
sys.path.append(str(project_root))

# Import existing pipeline components
from live_processor import LiveProcessor
from data_refiner.refiner import DataRefiner
from lmpc_checker.compliance_validator import ComplianceValidator
from config import DEVICE, DEFAULT_CAMERA_INDEX, EXPECTED_FIELDS

# Configure Streamlit page
st.set_page_config(
    page_title="Legal Metrology OCR Pipeline",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yaswanthsetty/legal-metrology-ocr-pipeline',
        'Report a bug': 'https://github.com/yaswanthsetty/legal-metrology-ocr-pipeline/issues',
        'About': """
        # Legal Metrology OCR Compliance Pipeline
        
        An intelligent, production-ready system for automated Legal Metrology 
        compliance validation using advanced Computer Vision and AI.
        
        **Version:** 2.0.0
        **License:** MIT
        **GitHub:** https://github.com/yaswanthsetty/legal-metrology-ocr-pipeline
        """
    }
)

# Custom CSS for professional styling
def load_css():
    """Load custom CSS styling"""
    css = """
    <style>
    /* Main app styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1f4037, #99f2c8);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f4037;
        margin-bottom: 1rem;
    }
    
    /* Status indicators */
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    /* Processing animation */
    .processing-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #1f4037;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa, #e9ecef);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #1f4037, #99f2c8);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Results panel */
    .results-panel {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    
    /* Compliance status */
    .compliance-compliant {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    
    .compliance-warning {
        background: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    
    .compliance-violation {
        background: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
        margin: 0.5rem 0;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []
    
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    
    if 'system_status' not in st.session_state:
        st.session_state.system_status = {
            'ocr_model_loaded': False,
            'refiner_model_loaded': False,
            'validator_loaded': False,
            'camera_available': False,
            'gpu_available': DEVICE == 'cuda'
        }

# System status checker
@st.cache_resource
def initialize_pipeline_components():
    """Initialize and cache pipeline components"""
    try:
        # Initialize OCR processor
        ocr_processor = LiveProcessor()
        st.session_state.system_status['ocr_model_loaded'] = True
        
        # Initialize data refiner
        data_refiner = DataRefiner()
        st.session_state.system_status['refiner_model_loaded'] = True
        
        # Initialize compliance validator
        compliance_validator = ComplianceValidator()
        st.session_state.system_status['validator_loaded'] = True
        
        return ocr_processor, data_refiner, compliance_validator
        
    except Exception as e:
        st.error(f"Failed to initialize pipeline components: {str(e)}")
        return None, None, None

def check_camera_availability():
    """Check if cameras are available"""
    try:
        import threading
        import pythoncom
        cameras = []
        error_msg = None
        
        def detect_cameras():
            nonlocal cameras, error_msg
            try:
                # Initialize COM in this thread
                pythoncom.CoInitialize()
                
                from pygrabber.dshow_graph import FilterGraph
                graph = FilterGraph()
                cameras = graph.get_input_devices()
                
                # Cleanup COM
                pythoncom.CoUninitialize()
            except Exception as e:
                error_msg = str(e)
                try:
                    pythoncom.CoUninitialize()
                except:
                    pass
        
        # Run camera detection in a separate thread with proper COM initialization
        thread = threading.Thread(target=detect_cameras)
        thread.start()
        thread.join(timeout=3)  # 3 second timeout
        
        if cameras:
            st.session_state.system_status['camera_available'] = True
            return cameras
        else:
            # Fallback camera detection using OpenCV
            import cv2
            for i in range(3):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    st.session_state.system_status['camera_available'] = True
                    cap.release()
                    return [f"Camera {i}"]
                cap.release()
            
            st.session_state.system_status['camera_available'] = False
            if error_msg:
                st.session_state.system_status['camera_error'] = error_msg
            return []
            
    except Exception as e:
        st.session_state.system_status['camera_available'] = False
        st.session_state.system_status['camera_error'] = str(e)
        return []

def create_system_status_panel():
    """Create system status panel"""
    st.sidebar.markdown("### 🖥️ System Status")
    
    status = st.session_state.system_status
    
    # OCR Model Status
    ocr_status = "✅ Loaded" if status['ocr_model_loaded'] else "❌ Not Loaded"
    st.sidebar.markdown(f"**OCR Model:** {ocr_status}")
    
    # Data Refiner Status
    refiner_status = "✅ Loaded" if status['refiner_model_loaded'] else "❌ Not Loaded"
    st.sidebar.markdown(f"**Data Refiner:** {refiner_status}")
    
    # Validator Status
    validator_status = "✅ Loaded" if status['validator_loaded'] else "❌ Not Loaded"
    st.sidebar.markdown(f"**Validator:** {validator_status}")
    
    # Camera Status
    camera_status = "✅ Available" if status['camera_available'] else "❌ Not Available"
    st.sidebar.markdown(f"**Camera:** {camera_status}")
    
    # GPU Status
    gpu_status = "✅ CUDA Available" if status['gpu_available'] else "💻 CPU Only"
    st.sidebar.markdown(f"**Processing:** {gpu_status}")

def create_dashboard_metrics():
    """Create dashboard metrics display"""
    col1, col2, col3, col4 = st.columns(4)
    
    # Total processed images
    total_processed = len(st.session_state.processing_history)
    col1.metric("📊 Total Processed", total_processed)
    
    # Compliance rate
    if total_processed > 0:
        compliant_count = sum(1 for item in st.session_state.processing_history 
                            if not item.get('violations', []))
        compliance_rate = (compliant_count / total_processed) * 100
        col2.metric("✅ Compliance Rate", f"{compliance_rate:.1f}%")
    else:
        col2.metric("✅ Compliance Rate", "0%")
    
    # Average processing time
    if total_processed > 0:
        avg_time = sum(item.get('processing_time', 0) for item in st.session_state.processing_history) / total_processed
        col3.metric("⏱️ Avg Processing Time", f"{avg_time:.1f}s")
    else:
        col3.metric("⏱️ Avg Processing Time", "0s")
    
    # System health
    all_loaded = all([
        st.session_state.system_status['ocr_model_loaded'],
        st.session_state.system_status['refiner_model_loaded'],
        st.session_state.system_status['validator_loaded']
    ])
    health_status = "🟢 Healthy" if all_loaded else "🟡 Partial"
    col4.metric("🏥 System Health", health_status)

def create_recent_activity_panel():
    """Create recent activity panel"""
    st.markdown("### 📋 Recent Activity")
    
    if not st.session_state.processing_history:
        st.info("No processing history yet. Start by capturing or uploading an image!")
        return
    
    # Show last 5 processed items
    recent_items = st.session_state.processing_history[-5:]
    
    for item in reversed(recent_items):
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            
            with col1:
                st.write(f"**{item.get('timestamp', 'Unknown')}**")
            
            with col2:
                st.write(f"Method: {item.get('method', 'Unknown')}")
            
            with col3:
                violations = item.get('violations', [])
                if not violations:
                    st.markdown('<span class="status-success">✅ Compliant</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="status-error">❌ {len(violations)} Violations</span>', unsafe_allow_html=True)
            
            with col4:
                st.write(f"{item.get('processing_time', 0):.1f}s")
            
            st.markdown("---")

def create_architecture_diagram():
    """Create system architecture visualization"""
    st.markdown("### 🏗️ System Architecture")
    
    # Create a simple architecture diagram using Plotly
    fig = go.Figure()
    
    # Define components
    components = [
        {"name": "📷 Camera/Upload", "x": 0, "y": 0, "color": "#ff7f0e"},
        {"name": "🔍 OCR Processing", "x": 1, "y": 0, "color": "#2ca02c"},
        {"name": "🛠️ Data Refinement", "x": 2, "y": 0, "color": "#d62728"},
        {"name": "⚖️ Compliance Check", "x": 3, "y": 0, "color": "#9467bd"},
        {"name": "📊 Report Generation", "x": 4, "y": 0, "color": "#8c564b"}
    ]
    
    # Add nodes
    for comp in components:
        fig.add_trace(go.Scatter(
            x=[comp["x"]], y=[comp["y"]],
            mode='markers+text',
            marker=dict(size=60, color=comp["color"]),
            text=[comp["name"]],
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            name=comp["name"],
            showlegend=False
        ))
    
    # Add arrows
    for i in range(len(components) - 1):
        fig.add_annotation(
            x=components[i+1]["x"], y=components[i+1]["y"],
            ax=components[i]["x"], ay=components[i]["y"],
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=3, arrowsize=1.5, arrowwidth=2, arrowcolor="gray"
        )
    
    fig.update_layout(
        title="Processing Pipeline Flow",
        showlegend=False,
        xaxis=dict(visible=False, range=[-0.5, 4.5]),
        yaxis=dict(visible=False, range=[-0.5, 0.5]),
        height=200,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function"""
    # Load custom CSS
    load_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ Legal Metrology OCR Compliance Pipeline</h1>
        <p>Advanced Computer Vision and AI for Automated Compliance Validation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize pipeline components
    with st.spinner("Initializing pipeline components..."):
        ocr_processor, data_refiner, compliance_validator = initialize_pipeline_components()
    
    # Check camera availability
    available_cameras = check_camera_availability()
    
    # Create sidebar with system status
    create_system_status_panel()
    
    # Navigation
    st.sidebar.markdown("### 🧭 Navigation")
    st.sidebar.markdown("""
    **📄 Pages Available:**
    - 📷 Live Camera - Real-time capture and processing
    - 📂 Upload Image - Upload and process images
    - 📊 Batch Process - Process multiple images
    - ⚙️ Settings - Configure system parameters
    """)
    
    # Main dashboard content
    st.markdown("## 📊 Dashboard Overview")
    
    # System status check
    if not all([ocr_processor, data_refiner, compliance_validator]):
        st.error("⚠️ Some pipeline components failed to initialize. Please check system requirements and try restarting the application.")
        st.stop()
    
    # Display metrics
    create_dashboard_metrics()
    
    # Display architecture diagram
    create_architecture_diagram()
    
    # Two column layout for main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Quick start section
        st.markdown("### 🚀 Quick Start")
        
        quick_col1, quick_col2 = st.columns(2)
        
        with quick_col1:
            if st.button("📷 Start Live Camera", use_container_width=True):
                st.switch_page("pages/01_📷_Live_Camera.py")
        
        with quick_col2:
            if st.button("📂 Upload Images", use_container_width=True):
                st.switch_page("pages/02_📂_Upload_Image.py")
        
        quick_col3, quick_col4 = st.columns(2)
        
        with quick_col3:
            if st.button("📊 Batch Processing", use_container_width=True):
                st.switch_page("pages/03_📊_Batch_Process.py")
        
        with quick_col4:
            if st.button("⚙️ Settings", use_container_width=True):
                st.switch_page("pages/04_⚙️_Settings.py")
        
        # Recent activity
        create_recent_activity_panel()
    
    with col2:
        # System information panel
        st.markdown("### ℹ️ System Information")
        
        info_data = {
            "Parameter": [
                "OCR Engine",
                "AI Model",
                "Validation Rules",
                "Supported Languages",
                "Processing Device",
                "Python Version",
                "Streamlit Version"
            ],
            "Value": [
                "Surya OCR v0.16.7",
                "Google Flan-T5",
                "13+ Legal Metrology Rules",
                "English, Hindi",
                DEVICE.upper(),
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                st.__version__
            ]
        }
        
        info_df = pd.DataFrame(info_data)
        st.dataframe(info_df, hide_index=True, use_container_width=True)
        
        # Performance metrics
        st.markdown("### 📈 Performance Metrics")
        
        if st.session_state.processing_history:
            # Processing time trend
            recent_times = [item.get('processing_time', 0) for item in st.session_state.processing_history[-10:]]
            
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(
                y=recent_times,
                mode='lines+markers',
                name='Processing Time',
                line=dict(color='#1f4037', width=3),
                marker=dict(size=8)
            ))
            
            fig_perf.update_layout(
                title="Recent Processing Times",
                yaxis_title="Time (seconds)",
                xaxis_title="Recent Processes",
                height=300,
                margin=dict(l=0, r=0, t=50, b=50)
            )
            
            st.plotly_chart(fig_perf, use_container_width=True)
        else:
            st.info("No performance data available yet.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d;">
        <p>Legal Metrology OCR Compliance Pipeline v2.0.0</p>
        <p>🇮🇳 Built for the Indian retail ecosystem • ⚖️ Ensuring Legal Metrology compliance</p>
        <p><a href="https://github.com/yaswanthsetty/legal-metrology-ocr-pipeline" target="_blank">GitHub Repository</a> | 
           <a href="https://github.com/yaswanthsetty/legal-metrology-ocr-pipeline/issues" target="_blank">Report Issues</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()