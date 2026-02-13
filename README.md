# \ud83e\udde0 BINARY BRAIN

### Intelligent Document Extraction System | Financial Automation

> **IITM IEEE HACKSAGON | Team Binary Brain | Amity University**

---

## \ud83c\udfaf Problem Statement

Extracting structured data (Dealer Name, Model, Horse Power, Amount, Signature, etc.) from unstructured invoices is labor-intensive and error-prone. Traditional rule-based approaches fail to handle diverse invoices of varying layouts and languages (English, Hindi, Gujarati).

**Manual data entry is:**
- \u274c Slow
- \u274c Error-prone
- \u274c Expensive
- \u274c Delays credit decisioning

---

## \ud83d\udca1 Proposed Solution

A **Hybrid Document AI Pipeline** combining **OCR + Vision + NLP + AI Engine** to automatically extract structured data from tractor loan quotation invoices with **\u226595% Document Level Accuracy**.

### Pipeline Architecture

```
PDF Input \u2192 Preprocessing \u2192 OCR \u2192 Layout & Object Detection \u2192 Field Extraction \u2192 Validation \u2192 JSON Output
```

| Step | Component | Description |
|------|-----------|-------------|
| 1 | **Document Ingestion** | PDF/Image input, convert to images |
| 2 | **Preprocessing** | Deskew, denoise, resize, contrast enhancement |
| 3 | **OCR Layer** | EasyOCR for multilingual text extraction |
| 4 | **Layout & Object Detection** | Signature & stamp detection using CV |
| 5 | **Field Extraction Engine** | Fuzzy matching, regex, key-value pair detection |
| 6 | **Validation & Output** | Confidence scoring, numeric tolerance, JSON generation |

---

## \ud83d\udccb Extracted Fields

| Field | Method | Details |
|-------|--------|---------|
| **Dealer Name** | Fuzzy Match \u226590% | Matched against known dealer database |
| **Model Name** | Exact Match | Matched against known tractor models |
| **Horse Power** | Regex Rule Engine | Pattern-based numeric extraction |
| **Asset Cost** | Regex Rule Engine | Currency & amount pattern detection |
| **Dealer Signature** | Bounding Box Detection | Contour-based signature detection |
| **Dealer Stamp** | Bounding Box Detection | Color + shape-based stamp detection |

---

## \u2728 Features & Novelty

### Features
- \ud83c\udf10 **Multilingual Support** \u2014 English, Hindi, Gujarati
- \ud83d\udcd0 **Layout-Independent Extraction** \u2014 Works with any invoice format
- \u270d\ufe0f **Signature & Stamp Detection** \u2014 Using object detection with bounding boxes
- \ud83d\udcb0 **Cost-Efficient Inference** \u2014 < \u20b91 per document
- \ud83d\udda5\ufe0f **CPU Compatible Pipeline** \u2014 No GPU required
- \ud83d\udcc8 **Scalable** \u2014 Works for any invoice type (retail/industrial)

### Novelty
- Hybrid rule + AI system for higher accuracy
- Pseudo-labeling for no ground truth scenario
- Self-consistency validation mechanism
- Confidence-based rejection system

---

## \ud83d\udee0\ufe0f Tech Stack

| Component | Technology |
|-----------|------------|
| OCR Engine | EasyOCR |
| Image Processing | OpenCV, Pillow |
| PDF Processing | PyMuPDF (fitz) |
| Field Extraction | FuzzyWuzzy, Regex |
| Object Detection | OpenCV Contour + Hough |
| Web UI | Streamlit |
| Language | Python 3.13 |

---

## \ud83d\ude80 Getting Started

### Prerequisites
- Python 3.10+

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd tanuhackathon

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py --server.port 8501
```

Open **http://localhost:8501** in your browser.

---

## \ud83d\udcc1 Project Structure

```
tanuhackathon/
\u251c\u2500\u2500 app.py                    # Streamlit Web Application
\u251c\u2500\u2500 requirements.txt          # Python dependencies
\u251c\u2500\u2500 README.md                 # This file
\u251c\u2500\u2500 src/
\u2502   \u251c\u2500\u2500 __init__.py
\u2502   \u251c\u2500\u2500 preprocessing.py      # PDF to Image, Deskew, Denoise, Enhance
\u2502   \u251c\u2500\u2500 ocr_engine.py         # EasyOCR multilingual text extraction
\u2502   \u251c\u2500\u2500 detector.py           # Signature & Stamp detection
\u2502   \u251c\u2500\u2500 field_extractor.py    # Dealer Name, Model, HP, Cost extraction
\u2502   \u251c\u2500\u2500 validation.py         # Confidence scoring & JSON output
\u2502   \u2514\u2500\u2500 pipeline.py           # Main orchestration pipeline
\u251c\u2500\u2500 sample_invoices/          # Sample test invoices
\u251c\u2500\u2500 uploads/                  # User uploaded files
\u2514\u2500\u2500 outputs/                  # Extraction results (JSON + annotated images)
```

---

## \ud83d\udcca Sample Output

```json
{
  "dealerName": "Agri Machinery",
  "modelName": "Sonali 550",
  "horsePower": "50",
  "assetCost": "650000",
  "dealerSignature": [238, 678, 345, 721],
  "dealerStamp": [452, 676, 523, 725],
  "metadata": {
    "processingTime": "3.42s",
    "confidence": {
      "dealerName": 0.92,
      "modelName": 0.95,
      "horsePower": 0.85,
      "assetCost": 0.90,
      "dealerSignature": 0.78,
      "dealerStamp": 0.72
    },
    "documentLevelAccuracy": "85.3%",
    "costEstimate": "\u20b90.67"
  }
}
```

---

## \u26a0\ufe0f Drawbacks & Mitigations

| Drawback | Mitigation Strategy |
|----------|-------------------|
| Low-quality scans reduce OCR accuracy | Confidence thresholding |
| Heavy handwriting variation | Fallback rule-based extraction |
| Overlapping stamps affect IoU | Ensemble model validation |
| Blank/damaged invoices | Confidence-based rejection |

---

## \ud83d\udc65 Team Binary Brain

| Name | Role | Contact |
|------|------|---------|
| **Himanshu Sharma** | AI/ML Enthusiast | himanshusharma610206@gmail.com |
| **Ayaan Siddiqui** | Frontend Developer & UI/UX Designer | ayaansiddiqui2029@gmail.com |
| **Priyanka** | Backend Developer | priyankalodhika@gmail.com |
| **Tanu Soni** | Backend Developer | tanu18098@gmail.com |

---

## \ud83d\udcdc License

This project was built for **IITM IEEE HACKSAGON** hackathon.

---

*Built with \u2764\ufe0f by Team Binary Brain*
