# 🧠 BINARY BRAIN

### Intelligent Document Extraction System | Financial Automation

> **IITM IEEE HACKSAGON | Team Binary Brain | Amity University**

---

## 🎯 Problem Statement

Extracting structured data (Dealer Name, Model, Horse Power, Amount, Signature, etc.) from unstructured invoices is labor-intensive and error-prone. Traditional rule-based approaches fail to handle diverse invoices of varying layouts and languages (English, Hindi, Gujarati).

**Manual data entry is:**
- ❌ Slow
- ❌ Error-prone
- ❌ Expensive
- ❌ Delays credit decisioning

---

## 💡 Proposed Solution

A **Hybrid Document AI Pipeline** combining **OCR + Vision + NLP + AI Engine** to automatically extract structured data from tractor loan quotation invoices with **≥95% Document Level Accuracy**.

### Pipeline Architecture

```
PDF Input → Preprocessing → OCR → Layout & Object Detection → Field Extraction → Validation → JSON Output
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

## 📋 Extracted Fields

| Field | Method | Details |
|-------|--------|---------|
| **Dealer Name** | Fuzzy Match ≥90% | Matched against known dealer database |
| **Model Name** | Exact Match | Matched against known tractor models |
| **Horse Power** | Regex Rule Engine | Pattern-based numeric extraction |
| **Asset Cost** | Regex Rule Engine | Currency & amount pattern detection |
| **Dealer Signature** | Bounding Box Detection | Contour-based signature detection |
| **Dealer Stamp** | Bounding Box Detection | Color + shape-based stamp detection |

---

## ✨ Features & Novelty

### Features
- 🌐 **Multilingual Support** — English, Hindi, Gujarati
- 📐 **Layout-Independent Extraction** — Works with any invoice format
- ✍️ **Signature & Stamp Detection** — Using object detection with bounding boxes
- 💰 **Cost-Efficient Inference** — < $0.01 per document
- 🖥️ **CPU Compatible Pipeline** — No GPU required
- 📈 **Scalable** — Works for any invoice type (retail/industrial)

### Novelty
- Hybrid rule + AI system for higher accuracy
- Pseudo-labeling for no ground truth scenario
- Self-consistency validation mechanism
- Confidence-based rejection system

---

## 🛠️ Tech Stack

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

## 🚀 Getting Started

### Prerequisites
- Python 3.10+

### Installation

```bash
# Clone the repository
git clone https://github.com/DivyanshRajSoni/binary-brain.git
cd binary-brain

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

## 📁 Project Structure

```
binary-brain/
├── app.py                    # Streamlit Web Application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # PDF to Image, Deskew, Denoise, Enhance
│   ├── ocr_engine.py         # EasyOCR multilingual text extraction
│   ├── detector.py           # Signature & Stamp detection
│   ├── field_extractor.py    # Dealer Name, Model, HP, Cost extraction
│   ├── validation.py         # Confidence scoring & JSON output
│   └── pipeline.py           # Main orchestration pipeline
├── sample_invoices/          # Sample test invoices
├── uploads/                  # User uploaded files
└── outputs/                  # Extraction results (JSON + annotated images)
```

---

## 📊 Sample Output

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
    "costEstimate": "$0.008"
  }
}
```

---

## ⚠️ Drawbacks & Mitigations

| Drawback | Mitigation Strategy |
|----------|-------------------|
| Low-quality scans reduce OCR accuracy | Confidence thresholding |
| Heavy handwriting variation | Fallback rule-based extraction |
| Overlapping stamps affect IoU | Ensemble model validation |
| Blank/damaged invoices | Confidence-based rejection |

---

## 👥 Team Binary Brain

| Name | Role | Contact |
|------|------|---------|
| **Himanshu Sharma** | AI/ML Enthusiast | himanshusharma610206@gmail.com |
| **Ayaan Siddiqui** | Frontend Developer & UI/UX Designer | ayaansiddiqui2029@gmail.com |
| **Priyanka** | Backend Developer | priyankalodhika@gmail.com |
| **Tanu Soni** | Backend Developer | tanu18098@gmail.com |

---

## 📜 License

This project was built for **IITM IEEE HACKSAGON** hackathon.

---

*Built with ❤️ by Team Binary Brain*
