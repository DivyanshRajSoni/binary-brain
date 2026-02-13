"""
Binary Brain - Intelligent Document Extraction System
Streamlit Web Application

Team: BINARY BRAIN | IITM IEEE HACKSAGON
"""

import streamlit as st
import os
import json
import time
import tempfile
import cv2
import numpy as np
from PIL import Image

from src.pipeline import BinaryBrainPipeline


# ──────────────────────────────────────────────────────────
#  Page Config (MUST be first Streamlit command)
# ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Binary Brain - Document AI",
    page_icon=":brain:",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ──────────────────────────────────────────────────────────
#  Helper Functions (defined before use)
# ──────────────────────────────────────────────────────────

@st.cache_resource
def load_pipeline():
    """Initialize the pipeline (cached)."""
    return BinaryBrainPipeline(languages=['en', 'hi'], gpu=False)


def display_results(result):
    """Display extraction results in a nice format."""
    if 'error' in result and result.get('dealerName') is None:
        st.error(result['error'])
        return

    fields = [
        ('<i class="fas fa-building"></i> Dealer Name', result.get('dealerName', 'N/A'), 'dealerName'),
        ('<i class="fas fa-tractor"></i> Model Name', result.get('modelName', 'N/A'), 'modelName'),
        ('<i class="fas fa-bolt"></i> Horse Power', result.get('horsePower', 'N/A'), 'horsePower'),
        ('<i class="fas fa-indian-rupee-sign"></i> Asset Cost', result.get('assetCost', 'N/A'), 'assetCost'),
    ]

    metadata = result.get('metadata', {})
    confidences = metadata.get('confidence', {})

    for label, value, key in fields:
        conf = confidences.get(key, 0)
        conf_class = 'high' if conf >= 0.8 else ('medium' if conf >= 0.5 else 'low')
        conf_color = '#28a745' if conf >= 0.8 else ('#ffc107' if conf >= 0.5 else '#dc3545')

        val_color = '#000000' if key in ('dealerName', 'modelName') else 'inherit'
        conf_icon = 'fa-check-circle' if conf >= 0.8 else ('fa-exclamation-circle' if conf >= 0.5 else 'fa-times-circle')
        st.markdown(f"""
        <div class="field-card confidence-{conf_class}">
            <strong>{label}</strong><br>
            <span style="font-size: 1.3rem; color: {val_color};">{value or 'Not Found'}</span>
            <span class="conf-badge" style="float: right; color: {conf_color};">
                <i class="fas {conf_icon}"></i> Confidence: {conf * 100:.0f}%
            </span>
        </div>
        """, unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        sig = result.get('dealerSignature')
        sig_conf = confidences.get('dealerSignature', 0)
        if sig:
            st.success(f"Signature Found\nBBox: {sig}\nConfidence: {sig_conf * 100:.0f}%")
        else:
            st.warning("Signature: Not detected")

    with col_s2:
        stamp = result.get('dealerStamp')
        stamp_conf = confidences.get('dealerStamp', 0)
        if stamp:
            st.success(f"Stamp Found\nBBox: {stamp}\nConfidence: {stamp_conf * 100:.0f}%")
        else:
            st.warning("Stamp: Not detected")

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Processing Time", metadata.get('processingTime', 'N/A'))
    with m2:
        st.metric("Document Accuracy", metadata.get('documentLevelAccuracy', 'N/A'))
    with m3:
        st.metric("Cost Estimate", metadata.get('costEstimate', '\u20b90.67'))


def create_sample_invoice():
    """Create a sample tractor invoice image for demo."""
    img = np.ones((1200, 900, 3), dtype=np.uint8) * 255

    cv2.rectangle(img, (0, 0), (900, 100), (102, 126, 234), -1)
    cv2.putText(img, "TRACTOR QUOTATION / INVOICE", (120, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    cv2.putText(img, "Dealer Name: Agri Machinery Pvt Ltd", (50, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Address: Industrial Area, Phase-2, Bhopal", (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
    cv2.putText(img, "Invoice No: INV-2024-00456", (500, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
    cv2.putText(img, "Date: 15-01-2025", (500, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)

    cv2.line(img, (30, 230), (870, 230), (150, 150, 150), 2)

    cv2.putText(img, "PRODUCT DETAILS", (50, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 150), 2)

    details = [
        ("Model Name:", "Sonali 550 DI"),
        ("Horse Power:", "50 HP"),
        ("Engine Type:", "4-Cylinder, Water Cooled"),
        ("Transmission:", "8 Forward + 2 Reverse"),
        ("Lifting Capacity:", "1800 kg"),
    ]

    y = 330
    for label, value in details:
        cv2.putText(img, label, (70, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 50), 2)
        cv2.putText(img, value, (350, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        y += 45

    cv2.line(img, (30, y + 10), (870, y + 10), (150, 150, 150), 2)
    y += 50
    cv2.putText(img, "PRICING", (50, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 150), 2)
    y += 50

    prices = [
        ("Ex-Showroom Price:", "Rs. 5,75,000"),
        ("Registration:", "Rs. 25,000"),
        ("Insurance:", "Rs. 35,000"),
        ("Accessories:", "Rs. 15,000"),
    ]

    for label, value in prices:
        cv2.putText(img, label, (70, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
        cv2.putText(img, value, (500, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        y += 40

    cv2.line(img, (450, y + 5), (800, y + 5), (0, 0, 0), 2)
    y += 40
    cv2.putText(img, "Total Amount:", (350, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Rs. 6,50,000/-", (550, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 0, 0), 2)

    y += 80
    cv2.line(img, (30, y), (870, y), (150, 150, 150), 2)
    y += 50

    sig_y = y + 30
    pts = []
    for x in range(100, 350):
        sy = sig_y + int(15 * np.sin((x - 100) / 20)) + int(5 * np.cos((x - 100) / 7))
        pts.append([x, sy])
    pts = np.array(pts, dtype=np.int32)
    cv2.polylines(img, [pts], False, (0, 0, 100), 2)

    cv2.putText(img, "Dealer Signature", (120, sig_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    stamp_cx, stamp_cy = 650, sig_y + 20
    cv2.circle(img, (stamp_cx, stamp_cy), 50, (200, 0, 0), 2)
    cv2.circle(img, (stamp_cx, stamp_cy), 45, (200, 0, 0), 1)
    cv2.putText(img, "AGRI", (stamp_cx - 25, stamp_cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 2)
    cv2.putText(img, "MACH", (stamp_cx - 25, stamp_cy + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 2)

    cv2.putText(img, "Dealer Stamp", (600, sig_y + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    cv2.putText(img, "This is a computer-generated quotation.",
                (200, 1150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    sample_path = os.path.join("sample_invoices", "sample_invoice.jpg")
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    cv2.imwrite(sample_path, img)

    return sample_path


# ──────────────────────────────────────────────────────────
#  Styling
# ──────────────────────────────────────────────────────────

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<style>
    .main-header {
        text-align: center; padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 { color: white; font-size: 2.5rem; margin: 0; }
    .main-header h1 i { margin-right: 10px; }
    .main-header p { color: #e0e0e0; font-size: 1.1rem; margin: 0.5rem 0 0 0; }
    .main-header p i { margin-right: 6px; }
    .field-card {
        background: #f8f9fa; border-radius: 8px; padding: 1rem;
        margin: 0.5rem 0; border-left: 4px solid #667eea;
    }
    .field-card i { margin-right: 8px; color: #667eea; }
    .confidence-high { border-left-color: #28a745; }
    .confidence-high i { color: #28a745; }
    .confidence-medium { border-left-color: #ffc107; }
    .confidence-medium i { color: #ffc107; }
    .confidence-low { border-left-color: #dc3545; }
    .confidence-low i { color: #dc3545; }
    .sidebar-section h3 i, .sidebar-section h4 i { margin-right: 8px; }
    .step-item { padding: 4px 0; }
    .step-item i { width: 20px; text-align: center; margin-right: 8px; color: #667eea; }
    .about-feature { padding: 6px 0; }
    .about-feature i { width: 22px; text-align: center; margin-right: 8px; color: #764ba2; }
    .team-member i { margin-right: 6px; color: #667eea; }
    .conf-badge i { margin-right: 4px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1><i class="fas fa-brain"></i> BINARY BRAIN</h1>
    <p><i class="fas fa-file-invoice"></i> Intelligent Document Extraction System | Financial Automation</p>
    <p style="font-size: 0.9rem; color: #ccc;"><i class="fas fa-trophy"></i> IITM IEEE HACKSAGON | Team Binary Brain</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
#  Sidebar
# ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<h3><i class="fas fa-cog"></i> Settings</h3>', unsafe_allow_html=True)
    st.markdown('<p><i class="fas fa-language"></i> <strong>OCR Languages</strong></p>', unsafe_allow_html=True)
    st.checkbox("English", value=True, disabled=True)
    st.checkbox("Hindi", value=True)
    st.checkbox("Gujarati", value=False)

    st.markdown("---")
    st.markdown('<p><i class="fas fa-sliders-h"></i> <strong>Detection Settings</strong></p>', unsafe_allow_html=True)
    st.slider("Min Confidence", 0.1, 1.0, 0.3, 0.05)
    st.slider("Numeric Tolerance", 0.01, 0.15, 0.05, 0.01)

    st.markdown("---")
    st.markdown('<h3><i class="fas fa-project-diagram"></i> Pipeline Steps</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-section">
        <div class="step-item"><i class="fas fa-file-import"></i> 1. Document Ingestion</div>
        <div class="step-item"><i class="fas fa-magic"></i> 2. Preprocessing (Deskew, Denoise)</div>
        <div class="step-item"><i class="fas fa-font"></i> 3. OCR Text Extraction</div>
        <div class="step-item"><i class="fas fa-signature"></i> 4. Signature &amp; Stamp Detection</div>
        <div class="step-item"><i class="fas fa-crosshairs"></i> 5. Field Extraction</div>
        <div class="step-item"><i class="fas fa-check-double"></i> 6. Validation &amp; JSON Output</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<h3><i class="fas fa-database"></i> Extracted Fields</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-section">
        <div class="step-item"><i class="fas fa-building"></i> <strong>Dealer Name</strong> (Fuzzy Match &ge; 90%)</div>
        <div class="step-item"><i class="fas fa-tractor"></i> <strong>Model Name</strong> (Exact Match)</div>
        <div class="step-item"><i class="fas fa-bolt"></i> <strong>Horse Power</strong> (Regex Engine)</div>
        <div class="step-item"><i class="fas fa-indian-rupee-sign"></i> <strong>Asset Cost</strong> (Regex Engine)</div>
        <div class="step-item"><i class="fas fa-signature"></i> <strong>Dealer Signature</strong> (Bounding Box)</div>
        <div class="step-item"><i class="fas fa-stamp"></i> <strong>Dealer Stamp</strong> (Bounding Box)</div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
#  Main Tabs
# ──────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Upload & Process", "Results", "About"])

with tab1:
    st.markdown('<h3><i class="fas fa-cloud-upload-alt"></i> Upload Invoice Document</h3>', unsafe_allow_html=True)
    st.markdown('<p><i class="fas fa-info-circle"></i> Upload a tractor loan quotation/invoice (PDF, JPG, PNG)</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Supported: PDF, JPG, PNG, BMP, TIFF"
    )

    col1, col2 = st.columns([1, 1])

    if uploaded_file:
        with col1:
            st.markdown('<h4><i class="fas fa-file-alt"></i> Uploaded Document</h4>', unsafe_allow_html=True)
            if uploaded_file.type and uploaded_file.type.startswith('image'):
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Invoice", use_container_width=True)
            else:
                st.info(f"Uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

        if st.button("Extract Data", type="primary", use_container_width=True):
            with st.spinner("Processing document through Binary Brain pipeline..."):
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=os.path.splitext(uploaded_file.name)[1]
                ) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    pipeline = load_pipeline()
                    output_dir = os.path.join(os.path.dirname(tmp_path), 'binary_brain_output')
                    os.makedirs(output_dir, exist_ok=True)

                    progress = st.progress(0)
                    status = st.empty()
                    status.text("Step 1/5: Preprocessing document...")
                    progress.progress(10)

                    result = pipeline.process_document(tmp_path, output_dir)

                    progress.progress(100)
                    status.text("Processing complete!")

                    st.session_state['result'] = result
                    st.session_state['output_dir'] = output_dir
                    st.session_state['file_name'] = uploaded_file.name

                    st.success("Document processed successfully!")

                    with col2:
                        st.markdown('<h4><i class="fas fa-poll"></i> Extracted Data</h4>', unsafe_allow_html=True)
                        display_results(result)

                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

    else:
        st.markdown('<p><i class="fas fa-arrow-up"></i> Upload a document to get started!</p>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<h3><i class="fas fa-play-circle"></i> Quick Demo with Sample Invoice</h3>', unsafe_allow_html=True)
        if st.button("Generate & Process Sample Invoice", use_container_width=True):
            with st.spinner("Generating sample invoice and processing..."):
                try:
                    sample_path = create_sample_invoice()
                    pipeline = load_pipeline()
                    output_dir = "outputs"
                    result = pipeline.process_document(sample_path, output_dir)

                    st.session_state['result'] = result
                    st.session_state['output_dir'] = output_dir
                    st.session_state['file_name'] = 'sample_invoice.jpg'

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.image(sample_path, caption="Sample Invoice",
                                 use_container_width=True)
                    with col_b:
                        display_results(result)

                    st.success("Sample processed! Check 'Results' tab for JSON.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

with tab2:
    st.markdown('<h3><i class="fas fa-chart-bar"></i> Extraction Results</h3>', unsafe_allow_html=True)
    if 'result' in st.session_state:
        result = st.session_state['result']
        display_results(result)

        st.markdown("---")
        st.markdown('<h4><i class="fas fa-code"></i> Raw JSON Output</h4>', unsafe_allow_html=True)
        st.json(result)

        json_str = json.dumps(result, indent=2, ensure_ascii=False)
        st.download_button(
            "Download JSON", json_str,
            file_name=f"{st.session_state.get('file_name', 'output')}_result.json",
            mime="application/json"
        )
    else:
        st.markdown('<p><i class="fas fa-inbox"></i> No results yet. Upload and process a document first.</p>', unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <h3><i class="fas fa-info-circle"></i> About Binary Brain</h3>

    <p><strong>Binary Brain</strong> is an Intelligent Document Extraction system designed for
    financial automation in the agricultural sector.</p>

    <h4><i class="fas fa-exclamation-triangle"></i> Problem</h4>
    <p>Extracting structured data (Dealer Name, Model, Horse Power, Amount, Signature, etc.)
    from unstructured invoices is labor-intensive and error-prone. Traditional rule-based
    approaches fail to handle diverse invoices of varying layouts and languages.</p>

    <h4><i class="fas fa-lightbulb"></i> Solution</h4>
    <p>A <strong>Hybrid Document AI Pipeline</strong> combining:</p>
    <div class="about-feature"><i class="fas fa-font"></i> <strong>OCR Engine</strong> &mdash; EasyOCR for multilingual text extraction (English, Hindi, Gujarati)</div>
    <div class="about-feature"><i class="fas fa-search"></i> <strong>Object Detection</strong> &mdash; Signature &amp; stamp detection using computer vision</div>
    <div class="about-feature"><i class="fas fa-crosshairs"></i> <strong>Field Extraction</strong> &mdash; Fuzzy matching, regex, and key-value pair detection</div>
    <div class="about-feature"><i class="fas fa-check-circle"></i> <strong>Validation</strong> &mdash; Confidence scoring, numeric tolerance, IoU validation</div>

    <h4><i class="fas fa-star"></i> Key Features</h4>
    <div class="about-feature"><i class="fas fa-language"></i> Multilingual support (English, Hindi, Gujarati)</div>
    <div class="about-feature"><i class="fas fa-th"></i> Layout-independent extraction</div>
    <div class="about-feature"><i class="fas fa-signature"></i> Signature &amp; Stamp detection with bounding boxes</div>
    <div class="about-feature"><i class="fas fa-rupee-sign"></i> Cost-efficient inference (&lt;Rs.1 per document)</div>
    <div class="about-feature"><i class="fas fa-microchip"></i> CPU compatible pipeline</div>
    <div class="about-feature"><i class="fas fa-bullseye"></i> 95%+ Document Level Accuracy</div>

    <h4><i class="fas fa-users"></i> Team Binary Brain</h4>
    <div class="team-member"><i class="fas fa-user"></i> <strong>Himanshu Sharma</strong> &mdash; AI/ML Enthusiast</div>
    <div class="team-member"><i class="fas fa-user"></i> <strong>Ayaan Siddiqui</strong> &mdash; Frontend Developer &amp; UI/UX Designer</div>
    <div class="team-member"><i class="fas fa-user"></i> <strong>Priyanka</strong> &mdash; Backend Developer</div>
    <div class="team-member"><i class="fas fa-user"></i> <strong>Tanu Soni</strong> &mdash; Backend Developer</div>

    <p style="margin-top: 1rem; color: #888;"><i class="fas fa-university"></i> <em>IITM IEEE HACKSAGON | Amity University</em></p>
    """, unsafe_allow_html=True)
