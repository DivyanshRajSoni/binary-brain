"""
Binary Brain - OCR Engine Module
Uses EasyOCR for multilingual text extraction (English, Hindi, Gujarati)
"""

import easyocr
import numpy as np
import cv2
from typing import List, Dict, Tuple


class OCREngine:
    """OCR Engine using EasyOCR for multilingual document text extraction."""

    def __init__(self, languages: list = None, gpu: bool = False):
        """
        Initialize OCR engine.
        Args:
            languages: List of language codes ['en', 'hi', 'gu']
            gpu: Whether to use GPU acceleration
        """
        if languages is None:
            languages = ['en', 'hi']
        self.languages = languages
        self.gpu = gpu
        self._reader = None

    @property
    def reader(self):
        """Lazy initialization of EasyOCR reader."""
        if self._reader is None:
            print(f"Initializing OCR engine with languages: {self.languages}")
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader

    def extract_text(self, image: np.ndarray, detail: int = 1) -> List[Dict]:
        """
        Extract text from image with bounding boxes and confidence.

        Args:
            image: Input image (BGR format)
            detail: 0 for text only, 1 for full details

        Returns:
            List of dicts with keys: text, bbox, confidence
        """
        results = self.reader.readtext(image)

        extracted = []
        for (bbox, text, confidence) in results:
            if confidence > 0.2:  # Filter low confidence
                # Convert bbox to [x1, y1, x2, y2] format
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x1, y1 = int(min(x_coords)), int(min(y_coords))
                x2, y2 = int(max(x_coords)), int(max(y_coords))

                extracted.append({
                    'text': text.strip(),
                    'bbox': [x1, y1, x2, y2],
                    'confidence': round(confidence, 4),
                    'raw_bbox': bbox
                })

        return extracted

    def extract_text_simple(self, image: np.ndarray) -> str:
        """Extract all text from image as a single string."""
        results = self.extract_text(image)
        return '\n'.join([r['text'] for r in results])

    def extract_with_layout(self, image: np.ndarray) -> Dict:
        """
        Extract text with layout information (line grouping).

        Returns:
            Dict with 'lines' (grouped text blocks) and 'all_results'
        """
        results = self.extract_text(image)
        if not results:
            return {'lines': [], 'all_results': []}

        # Sort by vertical position first, then horizontal
        results.sort(key=lambda r: (r['bbox'][1], r['bbox'][0]))

        # Group into lines based on vertical proximity
        lines = []
        current_line = [results[0]]
        line_threshold = 20  # pixels

        for i in range(1, len(results)):
            curr = results[i]
            prev = current_line[-1]

            # If y-position is similar, same line
            if abs(curr['bbox'][1] - prev['bbox'][1]) < line_threshold:
                current_line.append(curr)
            else:
                # Sort line items by x position
                current_line.sort(key=lambda r: r['bbox'][0])
                lines.append(current_line)
                current_line = [curr]

        if current_line:
            current_line.sort(key=lambda r: r['bbox'][0])
            lines.append(current_line)

        return {
            'lines': lines,
            'all_results': results
        }

    def get_text_near_label(self, results: List[Dict], label: str,
                            search_direction: str = 'right',
                            max_distance: int = 500) -> str:
        """
        Find text near a given label (key-value pair detection).

        Args:
            results: OCR results
            label: Label text to search for
            search_direction: 'right', 'below', or 'both'
            max_distance: Maximum pixel distance to search

        Returns:
            Value text found near the label
        """
        # Find the label in results
        label_lower = label.lower()
        label_result = None

        for r in results:
            if label_lower in r['text'].lower():
                label_result = r
                break

        if not label_result:
            return None

        label_bbox = label_result['bbox']
        candidates = []

        for r in results:
            if r == label_result:
                continue

            r_bbox = r['bbox']

            if search_direction in ['right', 'both']:
                # Check if to the right and roughly same vertical position
                if (r_bbox[0] > label_bbox[2] and
                        abs(r_bbox[1] - label_bbox[1]) < 30 and
                        r_bbox[0] - label_bbox[2] < max_distance):
                    distance = r_bbox[0] - label_bbox[2]
                    candidates.append((r['text'], distance))

            if search_direction in ['below', 'both']:
                # Check if below and roughly same horizontal position
                if (r_bbox[1] > label_bbox[3] and
                        abs(r_bbox[0] - label_bbox[0]) < 100 and
                        r_bbox[1] - label_bbox[3] < max_distance):
                    distance = r_bbox[1] - label_bbox[3]
                    candidates.append((r['text'], distance))

        if candidates:
            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]

        return None

    def visualize_results(self, image: np.ndarray,
                          results: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and text on image."""
        vis_image = image.copy()

        for r in results:
            bbox = r['bbox']
            text = r['text']
            conf = r['confidence']

            # Color based on confidence
            if conf > 0.8:
                color = (0, 255, 0)  # Green
            elif conf > 0.5:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red

            cv2.rectangle(vis_image, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]), color, 2)
            cv2.putText(vis_image, f"{text} ({conf:.2f})",
                        (bbox[0], bbox[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return vis_image
