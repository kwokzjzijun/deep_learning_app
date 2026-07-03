from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


_ELLIPSE_3X3 = (
    (-1, 0),
    (0, -1), (0, 0), (0, 1),
    (1, 0),
)


def _shift_view(padded: np.ndarray, dy: int, dx: int, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    return padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]


def _binary_dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        acc = np.zeros_like(out, dtype=bool)
        for dy, dx in _ELLIPSE_3X3:
            acc |= _shift_view(padded, dy, dx, out.shape)
        out = acc
    return out


def _binary_erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        acc = np.ones_like(out, dtype=bool)
        for dy, dx in _ELLIPSE_3X3:
            acc &= _shift_view(padded, dy, dx, out.shape)
        out = acc
    return out


def _smooth_3x3(gray: np.ndarray, iterations: int = 2) -> np.ndarray:
    out = gray.astype(np.float32, copy=True)
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="edge")
        out = (
            padded[:-2, :-2] + 2.0 * padded[:-2, 1:-1] + padded[:-2, 2:]
            + 2.0 * padded[1:-1, :-2] + 4.0 * padded[1:-1, 1:-1] + 2.0 * padded[1:-1, 2:]
            + padded[2:, :-2] + 2.0 * padded[2:, 1:-1] + padded[2:, 2:]
        ) / 16.0
    return out


def _make_gray_work(roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = roi.astype(np.float32).mean(axis=2)
    if np.any(mask):
        fill_val = float(np.median(gray[mask]))
    else:
        fill_val = float(np.median(gray))

    work = gray.copy()
    work[~_binary_dilate(mask, iterations=2)] = fill_val
    return _smooth_3x3(work, iterations=2)


def _circle_metrics(mask: np.ndarray, cx: int, cy: int, radius: int) -> Tuple[float, float]:
    h, w = mask.shape
    yy, xx = np.ogrid[:h, :w]
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    inter = int(np.count_nonzero(circle & mask))
    mask_area = max(1, int(np.count_nonzero(mask)))
    circle_area = max(1, int(np.count_nonzero(circle)))
    return inter / float(mask_area), inter / float(circle_area)


def _estimate_radius(
    grad: np.ndarray,
    edge_mask: np.ndarray,
    cx: float,
    cy: float,
    r_min: int,
    r_max: int,
) -> Optional[Tuple[int, float]]:
    ys, xs = np.where(edge_mask)
    if xs.size < 24 or r_max < r_min:
        return None

    d = np.sqrt((xs.astype(np.float32) - cx) ** 2 + (ys.astype(np.float32) - cy) ** 2)
    valid = (d >= float(r_min)) & (d <= float(r_max))
    if int(np.count_nonzero(valid)) < 16:
        return None

    bins = np.arange(r_min, r_max + 2, dtype=np.int32)
    weights = grad[ys[valid], xs[valid]].astype(np.float32)
    hist, _ = np.histogram(d[valid], bins=bins, weights=weights)
    if hist.size == 0:
        return None

    smooth = np.convolve(hist.astype(np.float32), np.array([1, 2, 3, 2, 1], dtype=np.float32), mode="same")
    peak_idx = int(np.argmax(smooth))
    radius = int(bins[peak_idx])
    strength = float(smooth[peak_idx] / max(1, int(np.count_nonzero(valid))))
    return radius, strength


def _find_refined_circle(mask: np.ndarray, roi: np.ndarray) -> Optional[Tuple[int, int, int]]:
    ys, xs = np.where(mask)
    if xs.size < 16:
        return None

    h, w = mask.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    mask_w = x1 - x0 + 1
    mask_h = y1 - y0 + 1
    base_r = max(2.0, min(max(mask_w, mask_h) / 2.0, min(h, w) / 2.0))

    bbox_cx = (x0 + x1) / 2.0
    bbox_cy = (y0 + y1) / 2.0
    cen_cx = float(xs.mean())
    cen_cy = float(ys.mean())
    delta = max(2.0, round(base_r * 0.03))

    centers = (
        (bbox_cx, bbox_cy),
        (cen_cx, cen_cy),
        (bbox_cx - delta, bbox_cy),
        (bbox_cx + delta, bbox_cy),
        (bbox_cx, bbox_cy - delta),
        (bbox_cx, bbox_cy + delta),
    )

    gray_work = _make_gray_work(roi, mask)
    gy, gx = np.gradient(gray_work)
    grad = np.hypot(gx, gy)
    edge_region = _binary_dilate(mask, iterations=2)
    region_vals = grad[edge_region]
    if region_vals.size == 0:
        return None

    threshold = float(np.percentile(region_vals, 90))
    edge_mask = (grad >= threshold) & edge_region

    r_min = max(2, int(round(base_r * 0.80)))
    r_max = min(min(h, w) // 2, int(round(base_r * 1.20)))
    if r_max < r_min:
        r_max = r_min

    expected_fill = float(np.clip(xs.size / max(1.0, math.pi * (base_r ** 2)), 0.45, 0.95))
    best: Optional[Tuple[float, int, int, int]] = None

    for cx_f, cy_f in centers:
        if not (0 <= cx_f < w and 0 <= cy_f < h):
            continue

        est = _estimate_radius(grad, edge_mask, cx_f, cy_f, r_min, r_max)
        if est is None:
            radius = int(round(base_r))
            edge_strength = 0.0
        else:
            radius, edge_strength = est

        cx = int(round(cx_f))
        cy = int(round(cy_f))
        radius = max(2, min(radius, cx, cy, w - 1 - cx, h - 1 - cy))
        if radius < 2:
            continue

        mask_cover, inside_ratio = _circle_metrics(mask, cx, cy, radius)
        fill_consistency = 1.0 - min(1.0, abs(inside_ratio - expected_fill))
        center_penalty = (abs(cx_f - bbox_cx) + abs(cy_f - bbox_cy)) / max(1.0, base_r)
        score = 0.55 * edge_strength + 0.30 * mask_cover + 0.15 * fill_consistency - 0.08 * center_penalty

        if best is None or score > best[0]:
            best = (score, cx, cy, radius)

    if best is None:
        cx = int(round(bbox_cx))
        cy = int(round(bbox_cy))
        radius = int(round(base_r))
    else:
        _, cx, cy, radius = best

    radius = int(max(2, radius - 1))
    radius = max(2, min(radius, cx, cy, w - 1 - cx, h - 1 - cy))
    return (cx, cy, radius) if radius >= 2 else None


def _make_circle_crop(roi: np.ndarray, circle: Tuple[int, int, int]) -> Optional[np.ndarray]:
    cx, cy, radius = circle
    h, w = roi.shape[:2]
    radius = min(radius, cx, cy, w - 1 - cx, h - 1 - cy)
    if radius < 2:
        return None

    left = cx - radius
    top = cy - radius
    size = radius * 2 + 1
    patch = roi[top:top + size, left:left + size]
    if patch.shape[0] != size or patch.shape[1] != size:
        return None

    yy, xx = np.ogrid[:size, :size]
    circle_mask = (xx - radius) ** 2 + (yy - radius) ** 2 <= radius ** 2
    out = np.zeros_like(patch)
    out[circle_mask] = patch[circle_mask]
    return out


def refine_circle_crop(roi_path: str, mask_path: str, out_path: str) -> str:
    try:
        roi = np.asarray(Image.open(roi_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    except Exception:
        return ""

    if roi.ndim != 3 or mask.ndim != 2 or roi.shape[:2] != mask.shape:
        return ""

    mask = _binary_dilate(mask, iterations=1)
    mask = _binary_erode(mask, iterations=1)
    mask = _binary_erode(mask, iterations=1)
    mask = _binary_dilate(mask, iterations=1)
    if int(np.count_nonzero(mask)) < 16:
        return ""

    circle = _find_refined_circle(mask, roi)
    if circle is None:
        return ""

    circle_crop = _make_circle_crop(roi, circle)
    if circle_crop is None:
        return ""

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        Image.fromarray(circle_crop, mode="RGB").save(out)
    except Exception:
        return ""
    return str(out)
