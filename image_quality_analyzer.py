"""
Image Quality Analyzer Module
Part of Patent-Eligible OCR Pipeline

Computes image quality metrics and applies document-aware preprocessing.
New in v3: deskew, illumination normalization, card detection, upscaling.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image


@dataclass
class QualityMetrics:
    contrast_ratio: float
    edge_density: float
    noise_level: float
    brightness: float
    sharpness: float


@dataclass
class PreprocessingRecommendation:
    apply_clahe: bool
    apply_sharpen: bool
    apply_denoise: bool
    increase_dpi: bool
    recommended_dpi: int


class ImageQualityAnalyzer:
    """
    Analyzes and enhances document images for maximum OCR and Vision accuracy.

    New capabilities (v3):
    - Full-image deskew using Hough line analysis
    - Perspective correction for ID cards photographed at an angle
    - Adaptive illumination normalization for phone camera photos
    - Resolution upscaling for low-res inputs
    - Smart binarization for degraded documents
    """

    def __init__(
        self,
        contrast_threshold: float = 2.0,
        edge_threshold: float = 0.3,
        noise_threshold: float = 0.15,
    ):
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.noise_threshold = noise_threshold

    # ─── Quality Metrics ──────────────────────────────────────────────────

    def compute_metrics(self, image_region: np.ndarray) -> QualityMetrics:
        gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY) if len(image_region.shape) == 3 else image_region
        return QualityMetrics(
            contrast_ratio=self._compute_contrast_ratio(gray),
            edge_density=self._compute_edge_density(gray),
            noise_level=self._compute_noise_level(gray),
            brightness=float(np.mean(gray) / 255.0),
            sharpness=self._compute_sharpness(gray),
        )

    def _compute_contrast_ratio(self, gray: np.ndarray) -> float:
        mn, mx = float(np.min(gray)), float(np.max(gray))
        return 0.0 if (mx + mn == 0 or mx == mn) else (mx - mn) / (mx + mn)

    def _compute_edge_density(self, gray: np.ndarray) -> float:
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.mean(np.sqrt(sx**2 + sy**2)) / 255.0)

    def _compute_noise_level(self, gray: np.ndarray) -> float:
        mean = cv2.blur(gray.astype(np.float32), (5, 5))
        sqr_mean = cv2.blur(gray.astype(np.float32) ** 2, (5, 5))
        return float(np.mean(np.maximum(sqr_mean - mean**2, 0)) / (255.0 ** 2))

    def _compute_sharpness(self, gray: np.ndarray) -> float:
        return float(np.var(cv2.Laplacian(gray, cv2.CV_64F)) / (255.0 ** 2))

    # ─── Preprocessing Recommendation ────────────────────────────────────

    def suggest_preprocessing(self, metrics: QualityMetrics) -> PreprocessingRecommendation:
        return PreprocessingRecommendation(
            apply_clahe=metrics.contrast_ratio < self.contrast_threshold,
            apply_sharpen=metrics.edge_density < self.edge_threshold,
            apply_denoise=metrics.noise_level > self.noise_threshold,
            increase_dpi=metrics.sharpness < 0.1,
            recommended_dpi=600 if metrics.sharpness < 0.1 else 300,
        )

    def apply_preprocessing(self, image: np.ndarray, recommendation: PreprocessingRecommendation) -> np.ndarray:
        processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        if recommendation.apply_clahe:
            processed = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(processed)
        if recommendation.apply_denoise:
            processed = cv2.fastNlMeansDenoising(processed, h=10)
        if recommendation.apply_sharpen:
            processed = cv2.filter2D(processed, -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))
        return processed

    # ─── New: Document-Level Enhancement Pipeline ─────────────────────────

    def enhance_document_image(self, pil_image: Image.Image) -> Image.Image:
        """
        Full document enhancement pipeline for maximum extraction accuracy.
        Applied once to the whole image before any extraction strategy.

        Steps:
        1. Upscale low-resolution images
        2. Normalize illumination (fixes phone camera uneven lighting)
        3. Deskew (correct tilt)
        4. Adaptive contrast enhancement
        """
        img = np.array(pil_image.convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Step 1: Upscale if too small (min 1200px on longest side for good Vision accuracy)
        img = self._upscale_if_needed(img, min_side=1200)

        # Step 2: Normalize illumination
        img = self._normalize_illumination(img)

        # Step 3: Deskew
        img = self._deskew(img)

        # Step 4: Adaptive per-channel CLAHE (enhances contrast without over-processing)
        img = self._apply_clahe_color(img)

        result = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return result

    def detect_and_crop_card(self, pil_image: Image.Image) -> Image.Image:
        """
        Detect a rectangular card/document in the image and perspective-correct it.
        Falls back to the original image if no clear card boundary is found.
        """
        img = np.array(pil_image.convert('RGB'))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        try:
            cropped = self._perspective_crop(img_bgr)
            if cropped is not None:
                h, w = cropped.shape[:2]
                # Only use crop if it's a reasonable card aspect ratio (width > height)
                if w > h * 0.9:
                    return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        except Exception:
            pass

        return pil_image

    # ─── Internal Enhancement Methods ─────────────────────────────────────

    def _upscale_if_needed(self, img: np.ndarray, min_side: int = 1200) -> np.ndarray:
        """Upscale image using Lanczos if either dimension is too small."""
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest >= min_side:
            return img
        scale = min_side / longest
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    def _normalize_illumination(self, img: np.ndarray) -> np.ndarray:
        """
        Correct uneven illumination using LAB color space normalization.
        Fixes the common phone-camera-on-card reflection problem.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        # CLAHE on L channel only
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(l_channel)

        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        """
        Correct document skew using Hough line angle detection.
        Handles tilts up to ±15 degrees.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=max(100, int(min(img.shape[:2]) * 0.2)))

        if lines is None or len(lines) < 3:
            return img

        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            if abs(angle) < 15:  # Only correct small tilts
                angles.append(angle)

        if not angles:
            return img

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:  # Skip trivial corrections
            return img

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def _apply_clahe_color(self, img: np.ndarray) -> np.ndarray:
        """Apply mild CLAHE to the luminance channel for contrast enhancement."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _perspective_crop(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the largest rectangular object (ID card) and apply perspective transform.
        Returns the warped card image, or None if no clear rectangle found.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 100)

        # Dilate to close small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edged = cv2.dilate(edged, kernel, iterations=1)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Sort by area, take largest
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        img_area = img.shape[0] * img.shape[1]
        for cnt in contours[:5]:
            area = cv2.contourArea(cnt)
            if area < img_area * 0.1:  # Card must be at least 10% of image
                break

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                warped = self._four_point_transform(img, pts)
                return warped

        return None

    @staticmethod
    def _four_point_transform(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Apply perspective warp given four corner points."""
        # Order: top-left, top-right, bottom-right, bottom-left
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1)
        rect = np.array([
            pts[np.argmin(s)],
            pts[np.argmin(d)],
            pts[np.argmax(s)],
            pts[np.argmax(d)],
        ], dtype=np.float32)

        tl, tr, br, bl = rect
        w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (w, h))
