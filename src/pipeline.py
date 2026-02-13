"""
Binary Brain - Main Processing Pipeline
Orchestrates the complete document processing workflow.
"""

import os
import time
import json
import cv2
import numpy as np

from src.preprocessing import preprocess_document
from src.ocr_engine import OCREngine
from src.detector import SignatureStampDetector
from src.field_extractor import FieldExtractor
from src.validation import ValidationEngine


class BinaryBrainPipeline:
    """Complete document AI pipeline for invoice data extraction."""

    def __init__(self, languages: list = None, gpu: bool = False):
        """
        Initialize the Binary Brain pipeline.

        Args:
            languages: OCR languages ['en', 'hi']
            gpu: Use GPU acceleration
        """
        self.ocr_engine = OCREngine(languages=languages or ['en', 'hi'], gpu=gpu)
        self.detector = SignatureStampDetector()
        self.field_extractor = FieldExtractor()
        self.validator = ValidationEngine()

    def process_document(self, file_path: str,
                         output_dir: str = None) -> dict:
        """
        Process a single document through the complete pipeline.

        Pipeline Steps:
        1. Document Ingestion & Preprocessing
        2. OCR Text Extraction
        3. Layout & Object Detection
        4. Field Extraction
        5. Validation & Output Generation

        Args:
            file_path: Path to PDF/image file
            output_dir: Directory to save outputs

        Returns:
            Structured JSON output with extracted fields
        """
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"  BINARY BRAIN - Document Processing Pipeline")
        print(f"{'='*60}")
        print(f"  Input: {os.path.basename(file_path)}")

        # ---- Step 1: Preprocessing ----
        print("\n[Step 1/5] Preprocessing document...")
        images = preprocess_document(file_path)

        if not images:
            return {
                'error': 'Could not read document',
                'dealerName': None,
                'modelName': None,
                'horsePower': None,
                'assetCost': None,
                'dealerSignature': None,
                'dealerStamp': None
            }

        print(f"  \u2713 Processed {len(images)} page(s)")

        # Process first page (main invoice)
        image = images[0]
        original_image = image.copy()

        # ---- Step 2: OCR ----
        print("\n[Step 2/5] Extracting text with OCR...")
        ocr_results = self.ocr_engine.extract_text(image)
        full_text = '\n'.join([r['text'] for r in ocr_results])
        print(f"  \u2713 Extracted {len(ocr_results)} text blocks")
        print(f"  \u2713 Total text length: {len(full_text)} characters")

        # ---- Step 3: Layout & Detection ----
        print("\n[Step 3/5] Detecting signatures & stamps...")
        detections = self.detector.detect_all(image)
        sig_count = len(detections['signatures'])
        stamp_count = len(detections['stamps'])
        print(f"  \u2713 Found {sig_count} signature(s), {stamp_count} stamp(s)")

        # ---- Step 4: Field Extraction ----
        print("\n[Step 4/5] Extracting structured fields...")
        extracted_fields = self.field_extractor.extract_all_fields(
            ocr_results, full_text
        )
        for field_name, field_data in extracted_fields.items():
            value = field_data.get('value', 'N/A')
            conf = field_data.get('confidence', 0)
            method = field_data.get('method', 'N/A')
            print(f"  \u2713 {field_name}: {value} (conf: {conf:.2f}, method: {method})")

        # ---- Step 5: Validation & Output ----
        print("\n[Step 5/5] Validating & generating output...")
        final_output = self.validator.validate_and_generate(
            extracted_fields, detections, start_time
        )

        # Save outputs if directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(file_path))[0]

            # Save JSON output
            json_path = os.path.join(output_dir, f"{base_name}_output.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, indent=2, ensure_ascii=False)
            print(f"  \u2713 Saved JSON: {json_path}")

            # Save annotated image
            annotated = self._create_annotated_image(
                original_image, ocr_results, detections, extracted_fields
            )
            img_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
            cv2.imwrite(img_path, annotated)
            print(f"  \u2713 Saved annotated image: {img_path}")

        doc_acc = final_output.get('metadata', {}).get('documentLevelAccuracy', 'N/A')
        proc_time = final_output.get('metadata', {}).get('processingTime', 'N/A')
        print(f"\n{'='*60}")
        print(f"  Document Accuracy: {doc_acc}")
        print(f"  Processing Time: {proc_time}")
        print(f"{'='*60}\n")

        return final_output

    def _create_annotated_image(self, image, ocr_results,
                                 detections, extracted_fields):
        """Create an annotated image showing all detections."""
        vis = image.copy()

        # Draw OCR text boxes (green)
        for r in ocr_results:
            bbox = r['bbox']
            cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                          (0, 200, 0), 1)

        # Draw signatures (red)
        for sig in detections.get('signatures', []):
            bbox = sig['bbox']
            cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                          (0, 0, 255), 3)
            cv2.putText(vis, f"SIGNATURE ({sig['confidence']:.2f})",
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw stamps (blue)
        for stamp in detections.get('stamps', []):
            bbox = stamp['bbox']
            cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                          (255, 0, 0), 3)
            cv2.putText(vis, f"STAMP ({stamp['confidence']:.2f})",
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Add result summary on image
        y_offset = 30
        for field_name, field_data in extracted_fields.items():
            value = field_data.get('value', 'N/A')
            text = f"{field_name}: {value}"
            cv2.putText(vis, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25

        return vis
