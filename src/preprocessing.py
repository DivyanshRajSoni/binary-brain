"""
Binary Brain - Preprocessing Module
Handles: PDF to Image conversion, Deskew, Denoise, Resize
"""

import os
import cv2
import numpy as np
from PIL import Image


def pdf_to_images(pdf_path: str, output_dir: str = None) -> list:
    """Convert PDF pages to images."""
    images = []

    # Handle image files directly
    ext = os.path.splitext(pdf_path)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
        img = cv2.imread(pdf_path)
        if img is not None:
            images.append(img)
        return images

    # For PDF files
    if ext == '.pdf':
        # Method 1: PyMuPDF (fitz) - no external dependency needed
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Render at 300 DPI (default is 72, so zoom = 300/72)
                zoom = 300 / 72
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                if pix.n == 4:  # RGBA
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                elif pix.n == 3:  # RGB
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = img_array
                images.append(img_bgr)
            doc.close()
            print(f"  [PyMuPDF] Converted {len(images)} pages from PDF")
            return images
        except Exception as e:
            print(f"  [PyMuPDF] Failed: {e}")

        # Method 2: pdf2image (requires Poppler)
        try:
            from pdf2image import convert_from_path
            pil_images = convert_from_path(pdf_path, dpi=300)
            for pil_img in pil_images:
                img_array = np.array(pil_img)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                images.append(img_bgr)
            print(f"  [pdf2image] Converted {len(images)} pages from PDF")
        except Exception as e:
            print(f"  [pdf2image] Failed: {e}")
            # Last fallback: try to read as image
            img = cv2.imread(pdf_path)
            if img is not None:
                images.append(img)

    return images


def deskew_image(image: np.ndarray) -> np.ndarray:
    """Correct image skew/rotation."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 10:
            return image

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Only correct small angles
        if abs(angle) > 10:
            return image

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception:
        return image


def denoise_image(image: np.ndarray) -> np.ndarray:
    """Remove noise from image."""
    try:
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        return denoised
    except Exception:
        return image


def enhance_image(image: np.ndarray) -> np.ndarray:
    """Enhance image contrast and sharpness for better OCR."""
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)

        # Merge back
        enhanced = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Sharpen
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        return sharpened
    except Exception:
        return image


def resize_image(image: np.ndarray, max_dimension: int = 2000) -> np.ndarray:
    """Resize image if too large while maintaining aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dimension:
        return image

    scale = max_dimension / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def preprocess_document(file_path: str) -> list:
    """
    Full preprocessing pipeline:
    1. Convert PDF/Image to images
    2. Deskew
    3. Denoise
    4. Enhance
    5. Resize
    """
    images = pdf_to_images(file_path)
    processed = []

    for img in images:
        # Step 1: Resize for processing
        img = resize_image(img)

        # Step 2: Deskew
        img = deskew_image(img)

        # Step 3: Denoise
        img = denoise_image(img)

        # Step 4: Enhance
        img = enhance_image(img)

        processed.append(img)

    return processed
