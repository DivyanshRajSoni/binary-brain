"""
Binary Brain - Signature & Stamp Detection Module
Uses contour detection and image analysis for detecting signatures and stamps.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple


class SignatureStampDetector:
    """Detects dealer signatures and stamps in invoice images."""

    def __init__(self):
        self.min_signature_area = 1000
        self.max_signature_area = 100000
        self.min_stamp_area = 2000
        self.max_stamp_area = 150000

    def detect_signatures(self, image: np.ndarray) -> List[Dict]:
        """
        Detect signature regions in the image.
        Signatures are typically dark ink strokes on light background.

        Returns:
            List of dicts with 'bbox', 'confidence', 'type'
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Focus on bottom half of document (signatures usually at bottom)
        roi_y_start = int(h * 0.4)
        roi = gray[roi_y_start:, :]

        # Adaptive threshold to detect ink strokes
        binary = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 10
        )

        # Morphological operations to connect signature strokes
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))

        dilated = cv2.dilate(binary, kernel_h, iterations=2)
        dilated = cv2.dilate(dilated, kernel_v, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        signatures = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_signature_area < area < self.max_signature_area:
                x, y, cw, ch = cv2.boundingRect(contour)
                aspect_ratio = cw / ch if ch > 0 else 0

                # Signatures are typically wider than tall
                if 1.2 < aspect_ratio < 8.0:
                    # Calculate stroke density
                    roi_region = binary[y:y + ch, x:x + cw]
                    density = np.sum(roi_region > 0) / (cw * ch) if cw * ch > 0 else 0

                    # Signatures have moderate density (not too dense like text blocks)
                    if 0.05 < density < 0.5:
                        confidence = min(0.95, 0.5 + density * 0.5 +
                                         (1.0 if 2.0 < aspect_ratio < 5.0 else 0.0) * 0.2)

                        # Adjust coordinates back to full image
                        signatures.append({
                            'bbox': [x, y + roi_y_start, x + cw, y + roi_y_start + ch],
                            'confidence': round(confidence, 4),
                            'type': 'signature',
                            'area': area
                        })

        # Sort by confidence and take top results
        signatures.sort(key=lambda s: s['confidence'], reverse=True)
        return signatures[:3]

    def detect_stamps(self, image: np.ndarray) -> List[Dict]:
        """
        Detect stamp regions in the image.
        Stamps are typically circular/rectangular colored regions.

        Returns:
            List of dicts with 'bbox', 'confidence', 'type'
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = gray.shape

        # Focus on bottom portion
        roi_y_start = int(h * 0.4)

        stamps = []

        # Method 1: Detect colored regions (stamps are often blue/red/purple)
        for color_name, lower, upper in [
            ('blue', np.array([100, 50, 50]), np.array([130, 255, 255])),
            ('red_low', np.array([0, 50, 50]), np.array([10, 255, 255])),
            ('red_high', np.array([160, 50, 50]), np.array([180, 255, 255])),
            ('purple', np.array([130, 50, 50]), np.array([160, 255, 255])),
        ]:
            mask = cv2.inRange(hsv[roi_y_start:, :], lower, upper)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            mask = cv2.dilate(mask, kernel, iterations=3)
            mask = cv2.erode(mask, kernel, iterations=1)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_stamp_area < area < self.max_stamp_area:
                    x, y, cw, ch = cv2.boundingRect(contour)
                    aspect_ratio = cw / ch if ch > 0 else 0

                    # Stamps are roughly square or circular
                    if 0.4 < aspect_ratio < 2.5:
                        # Check circularity
                        perimeter = cv2.arcLength(contour, True)
                        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

                        confidence = min(0.95, 0.4 + circularity * 0.3 +
                                         (0.2 if 0.7 < aspect_ratio < 1.4 else 0.0))

                        stamps.append({
                            'bbox': [x, y + roi_y_start, x + cw, y + roi_y_start + ch],
                            'confidence': round(confidence, 4),
                            'type': 'stamp',
                            'color': color_name,
                            'area': area
                        })

        # Method 2: Detect circular shapes (Hough circles)
        roi_gray = gray[roi_y_start:, :]
        blurred = cv2.GaussianBlur(roi_gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 1, 50,
            param1=100, param2=30,
            minRadius=30, maxRadius=150
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                cx, cy, r = circle
                stamps.append({
                    'bbox': [int(cx - r), int(cy + roi_y_start - r),
                             int(cx + r), int(cy + roi_y_start + r)],
                    'confidence': 0.7,
                    'type': 'stamp',
                    'color': 'detected_circle',
                    'area': int(np.pi * r * r)
                })

        # Remove duplicates (overlapping stamps)
        stamps = self._remove_overlapping(stamps)
        stamps.sort(key=lambda s: s['confidence'], reverse=True)
        return stamps[:3]

    def detect_all(self, image: np.ndarray) -> Dict:
        """Detect both signatures and stamps."""
        return {
            'signatures': self.detect_signatures(image),
            'stamps': self.detect_stamps(image)
        }

    def _remove_overlapping(self, detections: List[Dict],
                            iou_threshold: float = 0.3) -> List[Dict]:
        """Remove overlapping detections using IoU."""
        if len(detections) <= 1:
            return detections

        # Sort by confidence
        detections.sort(key=lambda d: d['confidence'], reverse=True)
        kept = []

        for det in detections:
            is_duplicate = False
            for existing in kept:
                iou = self._calculate_iou(det['bbox'], existing['bbox'])
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(det)

        return kept

    def _calculate_iou(self, bbox1: list, bbox2: list) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0
