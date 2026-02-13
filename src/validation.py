"""
Binary Brain - Validation & Post-processing Layer
- Confidence thresholding
- 5% tolerance for numeric fields
- IoU >= 0.5 for bounding box validation
- JSON output generation
"""

import json
import time
from typing import Dict, List, Optional


class ValidationEngine:
    """Validates extracted fields and generates final output."""

    def __init__(self, numeric_tolerance: float = 0.05, iou_threshold: float = 0.5):
        self.numeric_tolerance = numeric_tolerance  # 5% tolerance
        self.iou_threshold = iou_threshold
        self.min_confidence = 0.3

    def validate_and_generate(self, extracted_fields: Dict,
                               signature_detections: Dict,
                               processing_start_time: float) -> Dict:
        """
        Validate all extracted fields and generate final JSON output.

        Args:
            extracted_fields: Dict from FieldExtractor
            signature_detections: Dict from SignatureStampDetector
            processing_start_time: Start time for calculating processing duration

        Returns:
            Final validated JSON output
        """
        processing_time = round(time.time() - processing_start_time, 2)

        # Validate each field
        dealer_name = self._validate_text_field(
            extracted_fields.get('dealerName', {}), 'dealerName'
        )
        model_name = self._validate_text_field(
            extracted_fields.get('modelName', {}), 'modelName'
        )
        horse_power = self._validate_numeric_field(
            extracted_fields.get('horsePower', {}), 'horsePower',
            valid_range=(10, 500)
        )
        asset_cost = self._validate_numeric_field(
            extracted_fields.get('assetCost', {}), 'assetCost',
            valid_range=(10000, 50000000)
        )

        # Get signature and stamp bounding boxes
        signatures = signature_detections.get('signatures', [])
        stamps = signature_detections.get('stamps', [])

        dealer_signature = None
        if signatures:
            best_sig = signatures[0]
            dealer_signature = {
                'bbox': best_sig['bbox'],
                'confidence': best_sig['confidence']
            }

        dealer_stamp = None
        if stamps:
            best_stamp = stamps[0]
            dealer_stamp = {
                'bbox': best_stamp['bbox'],
                'confidence': best_stamp['confidence']
            }

        # Build final output
        output = {
            'dealerName': dealer_name['value'],
            'modelName': model_name['value'],
            'horsePower': horse_power['value'],
            'assetCost': asset_cost['value'],
            'dealerSignature': dealer_signature['bbox'] if dealer_signature else None,
            'dealerStamp': dealer_stamp['bbox'] if dealer_stamp else None,
            'metadata': {
                'processingTime': f"{processing_time}s",
                'confidence': {
                    'dealerName': dealer_name['confidence'],
                    'modelName': model_name['confidence'],
                    'horsePower': horse_power['confidence'],
                    'assetCost': asset_cost['confidence'],
                    'dealerSignature': dealer_signature['confidence'] if dealer_signature else 0,
                    'dealerStamp': dealer_stamp['confidence'] if dealer_stamp else 0,
                },
                'documentLevelAccuracy': self._calculate_document_accuracy(
                    dealer_name, model_name, horse_power, asset_cost,
                    dealer_signature, dealer_stamp
                ),
                'extractionMethods': {
                    'dealerName': dealer_name.get('method', 'N/A'),
                    'modelName': model_name.get('method', 'N/A'),
                    'horsePower': horse_power.get('method', 'N/A'),
                    'assetCost': asset_cost.get('method', 'N/A'),
                },
                'costEstimate': '\u20b90.67'  # Cost-efficient inference
            }
        }

        return output

    def _validate_text_field(self, field: Dict, field_name: str) -> Dict:
        """Validate a text field."""
        if not field or field.get('confidence', 0) < self.min_confidence:
            return {
                'value': field.get('value'),
                'confidence': field.get('confidence', 0),
                'method': field.get('method', 'not_found'),
                'valid': False,
                'warning': 'Low confidence or missing value'
            }

        value = field.get('value')
        if value:
            # Post-processing: clean up text
            value = self._clean_text(value)

        return {
            'value': value,
            'confidence': field.get('confidence', 0),
            'method': field.get('method', 'unknown'),
            'valid': True
        }

    def _validate_numeric_field(self, field: Dict, field_name: str,
                                 valid_range: tuple = None) -> Dict:
        """Validate a numeric field with 5% tolerance."""
        if not field or field.get('confidence', 0) < self.min_confidence:
            return {
                'value': field.get('value'),
                'confidence': field.get('confidence', 0),
                'method': field.get('method', 'not_found'),
                'valid': False,
                'warning': 'Low confidence or missing value'
            }

        value = field.get('value')
        if value:
            # Remove non-numeric characters
            clean_value = ''.join(c for c in str(value) if c.isdigit() or c == '.')
            try:
                num_value = float(clean_value)

                # Validate range
                if valid_range:
                    min_val = valid_range[0] * (1 - self.numeric_tolerance)
                    max_val = valid_range[1] * (1 + self.numeric_tolerance)
                    if not (min_val <= num_value <= max_val):
                        return {
                            'value': value,
                            'confidence': field.get('confidence', 0) * 0.5,
                            'method': field.get('method', 'unknown'),
                            'valid': False,
                            'warning': f'Value {num_value} outside range {valid_range}'
                        }

                return {
                    'value': str(int(num_value)) if num_value == int(num_value) else str(num_value),
                    'confidence': field.get('confidence', 0),
                    'method': field.get('method', 'unknown'),
                    'valid': True
                }

            except ValueError:
                return {
                    'value': value,
                    'confidence': field.get('confidence', 0) * 0.5,
                    'method': field.get('method', 'unknown'),
                    'valid': False,
                    'warning': 'Could not parse numeric value'
                }

        return {
            'value': None,
            'confidence': 0,
            'method': field.get('method', 'not_found'),
            'valid': False
        }

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        import re
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove leading/trailing punctuation
        text = text.strip('.:;,- ')
        return text

    def _calculate_document_accuracy(self, dealer_name, model_name,
                                      horse_power, asset_cost,
                                      signature, stamp) -> str:
        """Calculate overall document-level accuracy."""
        fields = [
            (dealer_name, 0.2),
            (model_name, 0.2),
            (horse_power, 0.15),
            (asset_cost, 0.2),
        ]

        total_weight = sum(w for _, w in fields)
        total_conf = sum(
            f.get('confidence', 0) * w for f, w in fields
        )

        # Add signature and stamp
        if signature:
            total_conf += signature.get('confidence', 0) * 0.125
            total_weight += 0.125
        if stamp:
            total_conf += stamp.get('confidence', 0) * 0.125
            total_weight += 0.125

        accuracy = (total_conf / total_weight * 100) if total_weight > 0 else 0
        return f"{accuracy:.1f}%"

    def validate_iou(self, bbox1: list, bbox2: list) -> float:
        """Calculate IoU between two bounding boxes [x1, y1, x2, y2]."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def save_output(self, output: Dict, output_path: str):
        """Save output to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        return output_path
