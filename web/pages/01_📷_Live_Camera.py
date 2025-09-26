"""
Live Camera Processing Page - Legal Metrology OCR Pipeline

Real-time camera capture and OCR processing with compliance validation.
Provides intuitive interface for camera selection, live preview, and instant results.
"""

import streamlit as st
import cv2
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
import sys
from PIL import Image
import io
import base64

# Add project root to path  
project_root = Path(__file__).parent.parent.parent  # Go up two levels from web/pages folder
sys.path.append(str(project_root))

from live_processor import LiveProcessor
from data_refiner.refiner import DataRefiner
from lmpc_checker.compliance_validator import ComplianceValidator
from config import DEVICE, DEFAULT_CAMERA_INDEX

# Page configuration
st.set_page_config(
    page_title="Live Camera - Legal Metrology OCR",
    page_icon="📷",
    layout="wide"
)

# Custom CSS for camera page
def load_camera_css():
    """Load custom CSS for camera page"""
    css = """
    <style>
    .camera-container {
        border: 2px solid #1f4037;
        border-radius: 10px;
        padding: 1rem;
        background: #f8f9fa;
        margin: 1rem 0;
    }
    
    .capture-button {
        background: linear-gradient(45deg, #ff6b35, #f7931e);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 25px;
        font-size: 1.2rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.4);
    }
    
    .capture-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(255, 107, 53, 0.6);
    }
    
    .processing-status {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .result-card {
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    
    .confidence-bar {
        background: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        height: 20px;
        margin: 0.5rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff5722, #4caf50);
        transition: width 0.3s ease;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Initialize session state for camera page
def initialize_camera_session():
    """Initialize camera-specific session state"""
    if 'camera_initialized' not in st.session_state:
        st.session_state.camera_initialized = False
    
    if 'selected_camera' not in st.session_state:
        st.session_state.selected_camera = DEFAULT_CAMERA_INDEX
    
    if 'capture_result' not in st.session_state:
        st.session_state.capture_result = None
    
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

# Camera detection and selection
@st.cache_data
def get_available_cameras():
    """Get list of available cameras"""
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
                cameras = []
                try:
                    pythoncom.CoUninitialize()
                except:
                    pass
        
        # Run camera detection in a separate thread with proper COM initialization
        thread = threading.Thread(target=detect_cameras)
        thread.start()
        thread.join(timeout=5)  # 5 second timeout
        
        if error_msg:
            st.warning(f"pygrabber error: {error_msg}")
            st.info("Using fallback camera detection...")
            return get_cameras_fallback()
        
        if cameras:
            return cameras
        else:
            st.info("No cameras detected via pygrabber, trying fallback...")
            return get_cameras_fallback()
            
    except Exception as e:
        st.warning(f"Camera detection error: {str(e)}")
        return get_cameras_fallback()

def get_cameras_fallback():
    """Fallback camera detection by trying to open camera indices"""
    import cv2
    cameras = []
    
    st.info("Scanning for cameras manually...")
    for i in range(5):  # Test first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(f"Camera {i}")
            cap.release()
        else:
            break
    
    if not cameras:
        cameras = ["Camera 0"]  # Default fallback
    
    return cameras

def create_camera_selection_panel():
    """Create camera selection interface"""
    st.markdown("### 📹 Camera Selection")
    
    cameras = get_available_cameras()
    
    if not cameras:
        st.error("❌ No cameras detected. Please connect a camera and refresh the page.")
        return None
    
    # Camera selection dropdown
    camera_options = [f"Camera {i}: {name}" for i, name in enumerate(cameras)]
    selected_option = st.selectbox(
        "Select Camera Device:",
        camera_options,
        index=st.session_state.selected_camera if st.session_state.selected_camera < len(cameras) else 0
    )
    
    # Extract camera index
    camera_index = int(selected_option.split(":")[0].split(" ")[1])
    st.session_state.selected_camera = camera_index
    
    # Camera settings
    with st.expander("🔧 Camera Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            resolution = st.selectbox(
                "Resolution:",
                ["1280x720", "1920x1080", "640x480"],
                index=0
            )
            
        with col2:
            fps = st.selectbox(
                "Frame Rate:",
                [30, 25, 15, 10],
                index=0
            )
    
    return camera_index, resolution, fps

def create_processing_controls():
    """Create processing control panel"""
    st.markdown("### ⚙️ Processing Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        confidence_threshold = st.slider(
            "OCR Confidence Threshold:",
            min_value=0.1,
            max_value=1.0,
            value=0.4,
            step=0.1,
            help="Higher values require more confident text detection"
        )
    
    with col2:
        language_selection = st.multiselect(
            "Language Support:",
            ["en", "hi"],
            default=["en"],
            help="Select languages for OCR processing"
        )
    
    # Advanced options
    with st.expander("🔬 Advanced Options"):
        enable_preprocessing = st.checkbox(
            "Enable Image Preprocessing",
            value=True,
            help="Apply image enhancement before OCR"
        )
        
        save_processed_images = st.checkbox(
            "Save Processed Images",
            value=False,
            help="Save captured and processed images to disk"
        )
        
        enable_gpu = st.checkbox(
            "Use GPU Acceleration",
            value=(DEVICE == 'cuda'),
            help="Use CUDA GPU for faster processing (if available)"
        )
    
    return {
        'confidence_threshold': confidence_threshold,
        'languages': language_selection,
        'enable_preprocessing': enable_preprocessing,
        'save_images': save_processed_images,
        'enable_gpu': enable_gpu
    }

def capture_and_process_image(camera_index: int, processing_options: dict):
    """Capture image from camera and process it through the pipeline"""
    
    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Use existing camera or initialize new one
        status_text.text("📷 Preparing camera...")
        progress_bar.progress(10)
        
        # Use existing camera from preview if available
        if ('camera_cap' in st.session_state and 
            st.session_state.camera_cap.isOpened()):
            cap = st.session_state.camera_cap
            release_after = False  # Don't release - keep for preview
        else:
            cap = cv2.VideoCapture(camera_index)
            release_after = True  # Release this temporary capture
            
        if not cap.isOpened():
            st.error("Failed to open camera. Please check camera connection.")
            return None
        
        # Step 2: Capture image
        status_text.text("📸 Capturing image...")
        progress_bar.progress(20)
        
        ret, frame = cap.read()
        
        # Only release if we created a temporary capture
        if release_after:
            cap.release()
        
        if not ret:
            st.error("Failed to capture image from camera.")
            return None
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Step 3: Use cached OCR processor or initialize new one
        status_text.text("🔍 Initializing OCR processor...")
        progress_bar.progress(30)
        
        # Use cached processor if available to avoid reloading models
        if 'ocr_processor' not in st.session_state:
            st.session_state.ocr_processor = LiveProcessor()
        
        ocr_processor = st.session_state.ocr_processor
        
        # Step 4: Process with OCR
        status_text.text("🔍 Extracting text with OCR...")
        progress_bar.progress(50)
        
        start_time = time.time()
        
        # Save temporary image for processing
        temp_image = Image.fromarray(frame_rgb)
        temp_path = "temp_capture.jpg"
        temp_image.save(temp_path)
        
        # Process image
        ocr_result = ocr_processor.process_single_image(temp_path)
        
        # Step 5: Data refinement
        status_text.text("🛠️ Refining extracted data...")
        progress_bar.progress(70)
        
        data_refiner = DataRefiner()
        refined_data = data_refiner.refine(ocr_result)
        
        # Step 6: Compliance validation
        status_text.text("⚖️ Validating compliance...")
        progress_bar.progress(90)
        
        compliance_validator = ComplianceValidator()
        violations = compliance_validator.validate(refined_data)
        
        # Step 7: Complete processing
        status_text.text("✅ Processing complete!")
        progress_bar.progress(100)
        
        processing_time = time.time() - start_time
        
        # Compile results
        result = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'method': 'Live Camera',
            'camera_index': camera_index,
            'processing_time': processing_time,
            'original_image': frame_rgb,
            'ocr_result': ocr_result,
            'refined_data': refined_data,
            'violations': violations,
            'compliance_status': 'COMPLIANT' if not violations else 'NON_COMPLIANT',
            'processing_options': processing_options
        }
        
        # Save to processing history
        if 'processing_history' not in st.session_state:
            st.session_state.processing_history = []
        st.session_state.processing_history.append(result)
        
        # Clean up
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        time.sleep(0.5)  # Brief pause to show completion
        status_text.empty()
        progress_bar.empty()
        
        return result
        
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        return None

def display_processing_results(result: dict):
    """Display comprehensive processing results"""
    if not result:
        return
    
    st.markdown("## 📊 Processing Results")
    
    # Results summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("⏱️ Processing Time", f"{result['processing_time']:.2f}s")
    
    with col2:
        violations_count = len(result.get('violations', []))
        st.metric("⚖️ Violations Found", violations_count)
    
    with col3:
        compliance_status = result.get('compliance_status', 'UNKNOWN')
        status_color = "✅" if compliance_status == 'COMPLIANT' else "❌"
        st.metric("📋 Status", f"{status_color} {compliance_status}")
    
    with col4:
        extracted_fields = len([k for k, v in result.get('refined_data', {}).items() if v])
        st.metric("📝 Fields Extracted", extracted_fields)
    
    # Tabbed results display
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Captured Image", "🔍 OCR Results", "📋 Structured Data", "⚖️ Compliance Report"])
    
    with tab1:
        st.markdown("### 📷 Captured Image")
        if 'original_image' in result:
            st.image(result['original_image'], caption="Original Captured Image", use_container_width=True)
        
        # Image metadata
        with st.expander("📊 Image Information"):
            if 'original_image' in result:
                image_shape = result['original_image'].shape
                st.write(f"**Dimensions:** {image_shape[1]} × {image_shape[0]} pixels")
                st.write(f"**Channels:** {image_shape[2] if len(image_shape) > 2 else 1}")
                st.write(f"**Camera Index:** {result.get('camera_index', 'Unknown')}")
                st.write(f"**Capture Time:** {result.get('timestamp', 'Unknown')}")
    
    with tab2:
        st.markdown("### 🔍 OCR Extraction Results")
        
        ocr_result = result.get('ocr_result', {})
        
        if isinstance(ocr_result, dict) and 'contents' in ocr_result:
            st.text_area(
                "Raw Extracted Text:",
                ocr_result['contents'],
                height=200,
                help="Raw text extracted by OCR engine"
            )
        else:
            st.text_area(
                "Raw Extracted Text:",
                str(ocr_result),
                height=200
            )
        
        # OCR confidence (if available)
        if isinstance(ocr_result, dict):
            confidence_data = ocr_result.get('confidence', None)
            if confidence_data:
                st.markdown("#### 📊 Detection Confidence")
                st.progress(confidence_data)
                st.write(f"Average confidence: {confidence_data:.2%}")
    
    with tab3:
        st.markdown("### 📋 Structured Product Data")
        
        refined_data = result.get('refined_data', {})
        
        if refined_data:
            # Display structured data in a nice format
            for field, value in refined_data.items():
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write(f"**{field.replace('_', ' ').title()}:**")
                with col2:
                    if value:
                        st.write(f"`{value}`")
                    else:
                        st.write("*Not found*")
            
            # JSON view
            with st.expander("🔧 Raw JSON Data"):
                st.json(refined_data)
        else:
            st.warning("No structured data was extracted from the image.")
    
    with tab4:
        st.markdown("### ⚖️ Legal Metrology Compliance Report")
        
        violations = result.get('violations', [])
        compliance_status = result.get('compliance_status', 'UNKNOWN')
        
        # Compliance status banner
        if compliance_status == 'COMPLIANT':
            st.success("🎉 **FULLY COMPLIANT** - This product meets all Legal Metrology requirements!")
        else:
            st.error(f"❌ **NON-COMPLIANT** - Found {len(violations)} compliance violations")
        
        # Violations list
        if violations:
            st.markdown("#### 📝 Compliance Violations")
            
            for i, violation in enumerate(violations, 1):
                severity = violation.get('severity', 'Unknown').upper()
                rule_id = violation.get('rule_id', 'UNKNOWN_RULE')
                description = violation.get('description', 'No description available')
                
                # Color coding based on severity
                if severity == 'CRITICAL':
                    st.error(f"**{i}. {rule_id}** (Critical)")
                elif severity == 'HIGH':
                    st.warning(f"**{i}. {rule_id}** (High)")
                else:
                    st.info(f"**{i}. {rule_id}** (Medium)")
                
                st.write(f"   {description}")
                st.markdown("---")
        
        # Compliance recommendations
        if violations:
            st.markdown("#### 💡 Recommendations")
            critical_violations = [v for v in violations if v.get('severity', '').upper() == 'CRITICAL']
            
            if critical_violations:
                st.error("🚨 **Critical issues must be resolved before product can be sold legally.**")
            
            st.info("📚 Refer to Legal Metrology (Packaged Commodities) Rules, 2011 for detailed requirements.")
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📋 Generate Detailed Report"):
                    st.info("Report generation feature coming soon!")
            
            with col2:
                if st.button("📧 Email Report"):
                    st.info("Email functionality coming soon!")
            
            with col3:
                if st.button("💾 Save Results"):
                    # Save results to session state for later access
                    st.session_state.saved_results = result
                    st.success("Results saved to session!")

def main():
    """Main function for live camera page"""    
    # Load custom CSS
    load_camera_css()
    
    # Initialize session state
    initialize_camera_session()
    
    # Cleanup function for page navigation
    def cleanup_resources():
        """Clean up camera resources when leaving page"""
        if 'camera_cap' in st.session_state and st.session_state.camera_cap.isOpened():
            st.session_state.camera_cap.release()
            del st.session_state.camera_cap
        st.session_state.camera_active = False
    
    # Register cleanup on page change
    if 'page_cleanup_registered' not in st.session_state:
        st.session_state.page_cleanup_registered = True
    
    # Page header
    st.markdown("""
    # 📷 Live Camera Processing
    
    Capture product labels in real-time and get instant compliance validation.
    Position your product label clearly in the camera view and press the capture button.
    """)
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 📹 Camera Controls")
        
        # Camera selection
        camera_result = create_camera_selection_panel()
        if not camera_result:
            st.stop()
        
        camera_index, resolution, fps = camera_result
        
        st.markdown("---")
        
        # Processing controls
        processing_options = create_processing_controls()
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📊 Session Stats")
        total_captures = len([h for h in st.session_state.get('processing_history', []) 
                            if h.get('method') == 'Live Camera'])
        st.metric("📸 Captures Today", total_captures)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Camera preview section
        st.markdown("### 📺 Camera Preview")
        
        # Camera feed placeholder
        camera_placeholder = st.empty()
        
        # Preview controls
        preview_col1, preview_col2, preview_col3 = st.columns(3)
        
        with preview_col1:
            if st.button("▶️ Start Preview", key="start_preview_btn", use_container_width=True):
                if camera_index is not None:
                    st.session_state.camera_active = True
                    st.success("Camera preview started!")
                    st.rerun()
                else:
                    st.error("No camera selected!")
        
        with preview_col2:
            if st.button("⏸️ Pause Preview", key="pause_preview_btn", use_container_width=True):
                st.session_state.camera_active = False
                # Clean up camera resources when pausing
                if 'camera_cap' in st.session_state and st.session_state.camera_cap.isOpened():
                    st.session_state.camera_cap.release()
                    del st.session_state.camera_cap
                st.info("Preview paused.")
                st.rerun()
        
        with preview_col3:
            if st.button("🔄 Refresh Camera", key="refresh_camera_btn", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        # Camera preview implementation
        if st.session_state.get('camera_active', False) and camera_index is not None:
            with camera_placeholder.container():
                try:
                    import cv2
                    import numpy as np
                    
                    # Initialize camera if not already done
                    if 'camera_cap' not in st.session_state or not st.session_state.camera_cap.isOpened():
                        st.session_state.camera_cap = cv2.VideoCapture(camera_index)
                        st.session_state.camera_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        st.session_state.camera_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        st.session_state.camera_cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    cap = st.session_state.camera_cap
                    
                    if not cap.isOpened():
                        st.error(f"Could not open camera {camera_index}")
                        st.session_state.camera_active = False
                    else:
                        # Capture current frame for preview
                        ret, frame = cap.read()
                        
                        if ret:
                            # Convert BGR to RGB for Streamlit
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            
                            # Display frame with container width
                            st.image(frame_rgb, 
                                   caption=f"📹 Live Preview - Camera {camera_index}", 
                                   use_container_width=True)
                            
                            st.success("📡 Live preview active - Camera ready for capture")
                            
                            # Auto-refresh every 100ms for smooth preview
                            time.sleep(0.1)
                            st.rerun()
                        else:
                            st.error("Failed to read from camera")
                            st.session_state.camera_active = False
                        
                except Exception as e:
                    st.error(f"Camera preview error: {str(e)}")
                    st.session_state.camera_active = False
        else:
            # Clean up camera when preview is stopped
            if 'camera_cap' in st.session_state and st.session_state.camera_cap.isOpened():
                st.session_state.camera_cap.release()
                del st.session_state.camera_cap
            
            # Show static placeholder when not active
            with camera_placeholder.container():
                if camera_index is not None:
                    st.info("📷 Camera ready. Click 'Start Preview' for live streaming.")
                    
                    # Show a static test capture
                    if st.button("📸 Test Camera", key="test_camera_btn"):
                        try:
                            import cv2
                            cap = cv2.VideoCapture(camera_index)
                            if cap.isOpened():
                                ret, frame = cap.read()
                                if ret:
                                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    st.image(frame_rgb, 
                                           caption=f"📸 Test Capture - Camera {camera_index}", 
                                           use_container_width=True)
                                    st.success("✅ Camera test successful!")
                                else:
                                    st.error("❌ Failed to capture test image")
                                cap.release()
                            else:
                                st.error("❌ Could not open camera")
                        except Exception as e:
                            st.error(f"Camera test error: {str(e)}")
                else:
                    st.warning("⚠️ Please select a camera first")
        
        # Capture section
        st.markdown("### 📸 Image Capture")
        
        capture_col1, capture_col2 = st.columns([2, 1])
        
        with capture_col1:
            if st.button("📷 CAPTURE IMAGE", type="primary", use_container_width=True):
                with st.spinner("Processing image through pipeline..."):
                    result = capture_and_process_image(camera_index, processing_options)
                    if result:
                        st.session_state.capture_result = result
                        st.session_state.processing_complete = True
                        st.success("✅ Image captured and processed successfully!")
                        st.rerun()
        
        with capture_col2:
            if st.button("🗑️ Clear Results", use_container_width=True):
                st.session_state.capture_result = None
                st.session_state.processing_complete = False
                st.rerun()
    
    with col2:
        # Instructions panel
        st.markdown("### 📖 Instructions")
        
        st.markdown("""
        **📋 How to use:**
        
        1. **Select Camera** - Choose your camera device
        2. **Adjust Settings** - Configure OCR parameters
        3. **Position Product** - Place label clearly in view
        4. **Capture Image** - Click the capture button
        5. **Review Results** - Check compliance status
        
        **💡 Tips for best results:**
        - Ensure good lighting
        - Keep text horizontal
        - Avoid glare and shadows
        - Use high contrast backgrounds
        - Hold camera steady
        """)
        
        # System status
        st.markdown("### 🖥️ System Status")
        
        status_container = st.container()
        with status_container:
            # Check system components
            try:
                # Test OCR processor
                LiveProcessor()
                st.success("✅ OCR Engine Ready")
            except:
                st.error("❌ OCR Engine Failed")
            
            try:
                # Test data refiner
                DataRefiner()
                st.success("✅ Data Refiner Ready")
            except:
                st.error("❌ Data Refiner Failed")
            
            try:
                # Test compliance validator
                ComplianceValidator()
                st.success("✅ Validator Ready")
            except:
                st.error("❌ Validator Failed")
            
            # GPU status
            if DEVICE == 'cuda':
                st.success("✅ GPU Acceleration")
            else:
                st.info("💻 CPU Processing")
    
    # Results display
    if st.session_state.processing_complete and st.session_state.capture_result:
        st.markdown("---")
        display_processing_results(st.session_state.capture_result)
    
    # Navigation
    st.markdown("---")
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    
    with nav_col1:
        if st.button("🏠 Back to Dashboard", use_container_width=True):
            st.switch_page("streamlit_app.py")
    
    with nav_col2:
        if st.button("📂 Upload Images", use_container_width=True):
            st.switch_page("pages/02_📂_Upload_Image.py")
    
    with nav_col3:
        if st.button("📊 Batch Processing", use_container_width=True):
            st.switch_page("pages/03_📊_Batch_Process.py")

if __name__ == "__main__":
    main()