"""
Batch Processing Page - Legal Metrology OCR Pipeline

Advanced batch processing with analytics, queuing, and comprehensive reporting.
Handles large volumes of images with progress tracking and detailed analytics.
"""

import streamlit as st
import os
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = Path(__file__).parent.parent.parent  # Go up two levels from web/pages folder
sys.path.append(str(project_root))

from live_processor import LiveProcessor
from data_refiner.refiner import DataRefiner
from lmpc_checker.compliance_validator import ComplianceValidator
from config import EXPECTED_FIELDS

# Page configuration
st.set_page_config(
    page_title="Batch Processing - Legal Metrology OCR",
    page_icon="📊",
    layout="wide"
)

# Constants
MAX_CONCURRENT_JOBS = 3
MAX_QUEUE_SIZE = 100
SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png', 'bmp', 'webp']

def load_batch_css():
    """Load custom CSS for batch processing page"""
    css = """
    <style>
    .batch-header {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .queue-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .analytics-panel {
        background: linear-gradient(135deg, #ffecd2, #fcb69f);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        color: #333;
    }
    
    .job-status-running {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .job-status-completed {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .job-status-failed {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .progress-container {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .metric-large {
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
    
    .violation-heatmap {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def initialize_batch_session():
    """Initialize batch processing session state"""
    if 'batch_queue' not in st.session_state:
        st.session_state.batch_queue = []
    
    if 'processing_jobs' not in st.session_state:
        st.session_state.processing_jobs = {}
    
    if 'completed_jobs' not in st.session_state:
        st.session_state.completed_jobs = []
    
    if 'batch_analytics' not in st.session_state:
        st.session_state.batch_analytics = {
            'total_processed': 0,
            'total_violations': 0,
            'processing_times': [],
            'violation_trends': [],
            'compliance_rates': [],
            'daily_stats': {}
        }
    
    if 'active_processors' not in st.session_state:
        st.session_state.active_processors = 0

class BatchProcessor:
    """Handles batch processing of images with threading support"""
    
    def __init__(self):
        self.ocr_processor = None
        self.data_refiner = None
        self.compliance_validator = None
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize pipeline components"""
        try:
            self.ocr_processor = LiveProcessor()
            self.data_refiner = DataRefiner()
            self.compliance_validator = ComplianceValidator()
        except Exception as e:
            st.error(f"Failed to initialize components: {str(e)}")
    
    def process_single_file(self, file_path: str, job_id: str) -> Dict[str, Any]:
        """Process a single file and return results"""
        try:
            start_time = time.time()
            
            # OCR Processing
            ocr_result = self.ocr_processor.process_single_image(file_path)
            
            # Data Refinement
            refined_data = self.data_refiner.refine(ocr_result)
            
            # Compliance Validation
            violations = self.compliance_validator.validate(refined_data)
            
            processing_time = time.time() - start_time
            
            result = {
                'job_id': job_id,
                'file_path': file_path,
                'filename': os.path.basename(file_path),
                'timestamp': datetime.now().isoformat(),
                'processing_time': processing_time,
                'ocr_result': ocr_result,
                'refined_data': refined_data,
                'violations': violations,
                'compliance_status': 'COMPLIANT' if not violations else 'NON_COMPLIANT',
                'status': 'completed'
            }
            
            return result
            
        except Exception as e:
            return {
                'job_id': job_id,
                'file_path': file_path,
                'filename': os.path.basename(file_path),
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'status': 'failed'
            }

def create_batch_header():
    """Create the batch processing header"""
    st.markdown("""
    <div class="batch-header">
        <h1>📊 Advanced Batch Processing</h1>
        <p>Process multiple images with advanced queuing, analytics, and reporting</p>
        <p><strong>Concurrent Processing | Real-time Analytics | Comprehensive Reports</strong></p>
    </div>
    """, unsafe_allow_html=True)

def create_queue_management_panel():
    """Create queue management interface"""
    st.markdown("### 📋 Queue Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload for batch
        uploaded_files = st.file_uploader(
            "Add files to processing queue",
            type=SUPPORTED_FORMATS,
            accept_multiple_files=True,
            help="Upload multiple images to add to the processing queue"
        )
        
        if uploaded_files:
            if st.button("➕ Add to Queue", type="primary"):
                # Save uploaded files and add to queue
                queue_items = []
                for uploaded_file in uploaded_files:
                    # Save file temporarily
                    temp_path = f"temp_batch_{len(st.session_state.batch_queue)}_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    queue_item = {
                        'id': f"job_{int(time.time())}_{len(st.session_state.batch_queue)}",
                        'filename': uploaded_file.name,
                        'file_path': temp_path,
                        'file_size': uploaded_file.size,
                        'added_time': datetime.now(),
                        'status': 'queued'
                    }
                    queue_items.append(queue_item)
                
                st.session_state.batch_queue.extend(queue_items)
                st.success(f"✅ Added {len(queue_items)} files to queue")
                st.rerun()
    
    with col2:
        # Queue statistics
        total_queued = len(st.session_state.batch_queue)
        active_jobs = len(st.session_state.processing_jobs)
        completed_jobs = len(st.session_state.completed_jobs)
        
        st.metric("📁 Queued", total_queued)
        st.metric("⚙️ Processing", active_jobs)
        st.metric("✅ Completed", completed_jobs)

def display_queue_status():
    """Display current queue status"""
    if not st.session_state.batch_queue and not st.session_state.processing_jobs:
        st.info("📭 Queue is empty. Upload files to get started.")
        return
    
    st.markdown("### 📊 Queue Status")
    
    # Queue items
    if st.session_state.batch_queue:
        st.markdown("#### 📋 Pending Items")
        for item in st.session_state.batch_queue[:10]:  # Show first 10
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.write(f"📄 {item['filename']}")
            
            with col2:
                st.write(f"Size: {item['file_size'] / 1024:.1f} KB")
            
            with col3:
                st.write("⏳ Queued")
        
        if len(st.session_state.batch_queue) > 10:
            st.write(f"... and {len(st.session_state.batch_queue) - 10} more items")
    
    # Processing items
    if st.session_state.processing_jobs:
        st.markdown("#### ⚙️ Currently Processing")
        for job_id, job_info in st.session_state.processing_jobs.items():
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.write(f"📄 {job_info.get('filename', 'Unknown')}")
            
            with col2:
                start_time = job_info.get('start_time', datetime.now())
                elapsed = (datetime.now() - start_time).total_seconds()
                st.write(f"Running: {elapsed:.1f}s")
            
            with col3:
                st.write("🔄 Processing")

def start_batch_processing():
    """Start batch processing with threading"""
    if not st.session_state.batch_queue:
        st.warning("No items in queue to process")
        return
    
    # Limit concurrent processing
    available_slots = MAX_CONCURRENT_JOBS - len(st.session_state.processing_jobs)
    if available_slots <= 0:
        st.warning("Maximum concurrent jobs reached. Please wait for completion.")
        return
    
    # Start processing jobs
    processor = BatchProcessor()
    
    items_to_process = st.session_state.batch_queue[:available_slots]
    
    for item in items_to_process:
        job_id = item['id']
        
        # Move from queue to processing
        st.session_state.processing_jobs[job_id] = {
            'filename': item['filename'],
            'file_path': item['file_path'],
            'start_time': datetime.now(),
            'status': 'processing'
        }
        
        # Remove from queue
        st.session_state.batch_queue.remove(item)
        
        # Process in background (simulated - in real implementation, use proper threading)
        result = processor.process_single_file(item['file_path'], job_id)
        
        # Move to completed
        st.session_state.completed_jobs.append(result)
        del st.session_state.processing_jobs[job_id]
        
        # Update analytics
        update_analytics(result)
    
    st.success(f"Started processing {len(items_to_process)} items")

def update_analytics(result: Dict[str, Any]):
    """Update analytics with new result"""
    analytics = st.session_state.batch_analytics
    
    if result.get('status') == 'completed':
        analytics['total_processed'] += 1
        
        violations = result.get('violations', [])
        analytics['total_violations'] += len(violations)
        
        processing_time = result.get('processing_time', 0)
        analytics['processing_times'].append(processing_time)
        
        # Daily stats
        today = datetime.now().strftime('%Y-%m-%d')
        if today not in analytics['daily_stats']:
            analytics['daily_stats'][today] = {
                'processed': 0,
                'compliant': 0,
                'violations': 0
            }
        
        analytics['daily_stats'][today]['processed'] += 1
        if result.get('compliance_status') == 'COMPLIANT':
            analytics['daily_stats'][today]['compliant'] += 1
        analytics['daily_stats'][today]['violations'] += len(violations)

def create_analytics_dashboard():
    """Create comprehensive analytics dashboard"""
    analytics = st.session_state.batch_analytics
    
    if analytics['total_processed'] == 0:
        st.info("📊 Analytics will appear after processing some images")
        return
    
    st.markdown("### 📈 Processing Analytics")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📁 Total Processed",
            analytics['total_processed']
        )
    
    with col2:
        compliance_rate = 0
        if analytics['total_processed'] > 0:
            compliant_count = len([job for job in st.session_state.completed_jobs 
                                 if job.get('compliance_status') == 'COMPLIANT'])
            compliance_rate = (compliant_count / analytics['total_processed']) * 100
        
        st.metric(
            "✅ Compliance Rate",
            f"{compliance_rate:.1f}%"
        )
    
    with col3:
        avg_time = 0
        if analytics['processing_times']:
            avg_time = sum(analytics['processing_times']) / len(analytics['processing_times'])
        
        st.metric(
            "⏱️ Avg Processing Time",
            f"{avg_time:.2f}s"
        )
    
    with col4:
        st.metric(
            "⚠️ Total Violations",
            analytics['total_violations']
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Processing time trend
        if len(analytics['processing_times']) > 1:
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                y=analytics['processing_times'],
                mode='lines+markers',
                name='Processing Time',
                line=dict(color='#667eea', width=3)
            ))
            
            fig_time.update_layout(
                title="Processing Time Trend",
                yaxis_title="Time (seconds)",
                xaxis_title="Batch Order",
                height=300
            )
            
            st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        # Compliance distribution
        compliant_count = len([job for job in st.session_state.completed_jobs 
                             if job.get('compliance_status') == 'COMPLIANT'])
        non_compliant_count = analytics['total_processed'] - compliant_count
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Compliant', 'Non-Compliant'],
            values=[compliant_count, non_compliant_count],
            hole=0.4,
            colors=['#28a745', '#dc3545']
        )])
        
        fig_pie.update_layout(
            title="Compliance Distribution",
            height=300
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Daily statistics
    if analytics['daily_stats']:
        st.markdown("#### 📅 Daily Processing Statistics")
        
        daily_df = pd.DataFrame.from_dict(analytics['daily_stats'], orient='index')
        daily_df.index = pd.to_datetime(daily_df.index)
        daily_df = daily_df.sort_index()
        
        fig_daily = go.Figure()
        
        fig_daily.add_trace(go.Scatter(
            x=daily_df.index,
            y=daily_df['processed'],
            mode='lines+markers',
            name='Processed',
            line=dict(color='#667eea')
        ))
        
        fig_daily.add_trace(go.Scatter(
            x=daily_df.index,
            y=daily_df['compliant'],
            mode='lines+markers',
            name='Compliant',
            line=dict(color='#28a745')
        ))
        
        fig_daily.update_layout(
            title="Daily Processing Trends",
            yaxis_title="Count",
            xaxis_title="Date",
            height=400
        )
        
        st.plotly_chart(fig_daily, use_container_width=True)

def create_violation_analytics():
    """Create detailed violation analytics"""
    if not st.session_state.completed_jobs:
        return
    
    st.markdown("### ⚖️ Violation Analytics")
    
    # Collect violation data
    violation_counts = {}
    severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0}
    
    for job in st.session_state.completed_jobs:
        violations = job.get('violations', [])
        for violation in violations:
            rule_id = violation.get('rule_id', 'UNKNOWN')
            severity = violation.get('severity', 'MEDIUM').upper()
            
            violation_counts[rule_id] = violation_counts.get(rule_id, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    if not violation_counts:
        st.success("🎉 No violations found in processed images!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Most common violations
        violation_df = pd.DataFrame(
            list(violation_counts.items()),
            columns=['Rule ID', 'Count']
        ).sort_values('Count', ascending=False)
        
        fig_violations = px.bar(
            violation_df.head(10),
            x='Count',
            y='Rule ID',
            orientation='h',
            title="Most Common Violations",
            color='Count',
            color_continuous_scale='Reds'
        )
        
        fig_violations.update_layout(height=400)
        st.plotly_chart(fig_violations, use_container_width=True)
    
    with col2:
        # Severity distribution
        severity_df = pd.DataFrame(
            list(severity_counts.items()),
            columns=['Severity', 'Count']
        )
        
        colors = {'CRITICAL': '#dc3545', 'HIGH': '#ffc107', 'MEDIUM': '#17a2b8'}
        
        fig_severity = px.pie(
            severity_df,
            values='Count',
            names='Severity',
            title="Violation Severity Distribution",
            color='Severity',
            color_discrete_map=colors
        )
        
        fig_severity.update_layout(height=400)
        st.plotly_chart(fig_severity, use_container_width=True)

def display_completed_jobs():
    """Display completed jobs with detailed results"""
    if not st.session_state.completed_jobs:
        return
    
    st.markdown("### ✅ Completed Jobs")
    
    # Summary table
    table_data = []
    for job in st.session_state.completed_jobs:
        table_data.append({
            'Filename': job.get('filename', 'Unknown'),
            'Status': job.get('status', 'Unknown'),
            'Compliance': job.get('compliance_status', 'Unknown'),
            'Violations': len(job.get('violations', [])),
            'Processing Time': f"{job.get('processing_time', 0):.2f}s",
            'Timestamp': job.get('timestamp', 'Unknown')[:19]  # Remove microseconds
        })
    
    df = pd.DataFrame(table_data)
    
    # Display with selection
    selected_indices = st.dataframe(
        df,
        use_container_width=True,
        selection_mode="multi-row",
        on_select="rerun",
        key="completed_jobs_dataframe"
    )
    
    # Detailed view for selected jobs
    if hasattr(selected_indices, 'selection') and selected_indices.selection.rows:
        st.markdown("#### 🔍 Job Details")
        
        for idx in selected_indices.selection.rows:
            if idx < len(st.session_state.completed_jobs):
                job = st.session_state.completed_jobs[idx]
                
                with st.expander(f"📄 {job.get('filename', 'Unknown')} - Details", expanded=True):
                    
                    if job.get('status') == 'failed':
                        st.error(f"❌ Processing failed: {job.get('error', 'Unknown error')}")
                        continue
                    
                    # Compliance summary
                    violations = job.get('violations', [])
                    if not violations:
                        st.success("✅ **FULLY COMPLIANT** - No violations found")
                    else:
                        st.error(f"❌ **NON-COMPLIANT** - {len(violations)} violations found")
                        
                        for i, violation in enumerate(violations, 1):
                            severity = violation.get('severity', 'Unknown').upper()
                            rule_id = violation.get('rule_id', 'UNKNOWN_RULE')
                            description = violation.get('description', 'No description')
                            
                            if severity == 'CRITICAL':
                                st.error(f"**{i}. {rule_id}** (Critical)")
                            elif severity == 'HIGH':
                                st.warning(f"**{i}. {rule_id}** (High)")
                            else:
                                st.info(f"**{i}. {rule_id}** (Medium)")
                            
                            st.write(f"   {description}")

def main():
    """Main function for batch processing page"""
    # Load custom CSS
    load_batch_css()
    
    # Initialize session state
    initialize_batch_session()
    
    # Page header
    create_batch_header()
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### ⚙️ Batch Settings")
        
        max_concurrent = st.slider(
            "Max Concurrent Jobs:",
            min_value=1,
            max_value=5,
            value=MAX_CONCURRENT_JOBS,
            help="Number of images to process simultaneously"
        )
        
        auto_start = st.checkbox(
            "Auto-start processing",
            value=False,
            help="Automatically start processing when files are added"
        )
        
        save_intermediate = st.checkbox(
            "Save intermediate results",
            value=True,
            help="Save results after each job completion"
        )
        
        st.markdown("---")
        
        # Processing controls
        st.markdown("### 🎮 Controls")
        
        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            start_batch_processing()
            st.rerun()
        
        if st.button("⏸️ Pause Queue", use_container_width=True):
            st.info("Queue paused (feature in development)")
        
        if st.button("🗑️ Clear Queue", use_container_width=True):
            st.session_state.batch_queue = []
            st.success("Queue cleared")
            st.rerun()
        
        st.markdown("---")
        
        # Export options
        st.markdown("### 📤 Export Options")
        
        if st.button("📊 Export Analytics", use_container_width=True):
            if st.session_state.completed_jobs:
                analytics_data = json.dumps(st.session_state.batch_analytics, indent=2, default=str)
                st.download_button(
                    label="⬇️ Download Analytics JSON",
                    data=analytics_data,
                    file_name=f"batch_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("No analytics data available")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Queue management
        create_queue_management_panel()
        
        # Queue status
        display_queue_status()
        
        # Processing controls
        st.markdown("### 🎮 Processing Controls")
        
        control_col1, control_col2, control_col3 = st.columns(3)
        
        with control_col1:
            if st.button("▶️ Start Batch", use_container_width=True):
                start_batch_processing()
                st.rerun()
        
        with control_col2:
            if st.button("⏸️ Pause All", use_container_width=True):
                st.info("Pause functionality in development")
        
        with control_col3:
            if st.button("🔄 Refresh Status", use_container_width=True):
                st.rerun()
    
    with col2:
        # Real-time statistics
        st.markdown("### 📊 Real-time Stats")
        
        stats_container = st.container()
        with stats_container:
            st.metric("📋 Queue Size", len(st.session_state.batch_queue))
            st.metric("⚙️ Active Jobs", len(st.session_state.processing_jobs))
            st.metric("✅ Completed", len(st.session_state.completed_jobs))
            
            if st.session_state.completed_jobs:
                success_rate = len([j for j in st.session_state.completed_jobs 
                                 if j.get('status') == 'completed']) / len(st.session_state.completed_jobs) * 100
                st.metric("📈 Success Rate", f"{success_rate:.1f}%")
    
    # Analytics dashboard
    if st.session_state.completed_jobs:
        st.markdown("---")
        create_analytics_dashboard()
        
        # Violation analytics
        create_violation_analytics()
        
        # Completed jobs
        display_completed_jobs()
    
    # Navigation
    st.markdown("---")
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    
    with nav_col1:
        if st.button("🏠 Back to Dashboard", use_container_width=True):
            st.switch_page("streamlit_app.py")
    
    with nav_col2:
        if st.button("📷 Live Camera", use_container_width=True):
            st.switch_page("pages/01_📷_Live_Camera.py")
    
    with nav_col3:
        if st.button("📂 Upload Images", use_container_width=True):
            st.switch_page("pages/02_📂_Upload_Image.py")

if __name__ == "__main__":
    main()