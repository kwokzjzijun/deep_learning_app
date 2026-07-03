from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REF_STEP5_WIDTH = 2968.0
REF_STEP5_HEIGHT = 347.0

# Human-labeled centers provided by user (on step_05 image coordinate system).
HUMAN_ANNOTATED_CENTERS_RAW: list[tuple[float, float]] = [
    (145.0, 90.0),
    (339.0, 90.0),
    (533.0, 90.0),
    (733.0, 90.0),
    (927.0, 90.0),
    (1123.0, 90.0),
    (1314.0, 91.0),
    (1508.0, 93.0),
    (1702.0, 90.0),
    (1890.0, 90.0),
    (2093.0, 104.0),
    (2284.0, 101.0),
    (2466.0, 107.0),
    (2469.0, 101.0),
    (2654.0, 101.0),
    (2834.0, 101.0),
]
# Radius from user annotation: x=91 -> x=143 at y=90.
HUMAN_ANNOTATED_RADIUS_PX = 55.0


@dataclass
class CircleDetection:
    cx: float
    cy: float
    r: float
    confidence: float
    method: str


@dataclass
class CircleCandidate:
    cx: float
    cy: float
    r: float
    geom_score: float
    pre_color_score: float
    ref_color_score: float
    score: float
    confidence: float
    method: str
    h: float
    s: float
    v: float
    blue_dom: float
    sat_contrast: float
    val_contrast: float


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def write_image(path: Path, image: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def build_roi_montage(roi_paths: list[Path], cols: int = 5) -> np.ndarray | None:
    if not roi_paths:
        return None

    rois = []
    for p in roi_paths:
        img = cv2.imread(str(p))
        if img is not None:
            rois.append(img)
    if not rois:
        return None

    h = max(r.shape[0] for r in rois)
    w = max(r.shape[1] for r in rois)
    normalized = []
    for idx, roi in enumerate(rois, start=1):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        y0 = (h - roi.shape[0]) // 2
        x0 = (w - roi.shape[1]) // 2
        canvas[y0 : y0 + roi.shape[0], x0 : x0 + roi.shape[1]] = roi
        cv2.putText(
            canvas,
            f"{idx:02d}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        normalized.append(canvas)

    rows = int(np.ceil(len(normalized) / cols))
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, tile in enumerate(normalized):
        r = i // cols
        c = i % cols
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = tile
    return grid


def make_roi(
    rectified_bgr: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    size_scale: float = 2.20,
) -> tuple[np.ndarray, float]:
    """Crop square ROI and mask outside circle to black."""
    h, w = rectified_bgr.shape[:2]
    side = max(64, int(round(r * size_scale)))
    half = side // 2

    center_x = int(round(cx))
    center_y = int(round(cy))

    x0 = clamp(center_x - half, 0, max(0, w - side))
    y0 = clamp(center_y - half, 0, max(0, h - side))
    x1 = min(w, x0 + side)
    y1 = min(h, y0 + side)

    crop = rectified_bgr[y0:y1, x0:x1].copy()
    ch, cw = crop.shape[:2]
    if ch == 0 or cw == 0:
        raise RuntimeError("Empty ROI crop.")

    mask = np.zeros((ch, cw), dtype=np.uint8)
    local_cx = clamp(center_x - x0, 0, cw - 1)
    local_cy = clamp(center_y - y0, 0, ch - 1)
    draw_r = int(round(min(r * 0.96, min(ch, cw) / 2.0 - 1)))
    draw_r = max(6, draw_r)
    cv2.circle(mask, (local_cx, local_cy), draw_r, 255, -1)

    roi = np.zeros_like(crop)
    roi[mask > 0] = crop[mask > 0]
    mask_ratio = float(np.count_nonzero(mask) / float(ch * cw))
    return roi, mask_ratio


def _merge_near_points(points: list[tuple[float, float]], x_eps: float = 10.0, y_eps: float = 10.0) -> list[tuple[float, float]]:
    if not points:
        return []
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    merged: list[tuple[float, float]] = []
    for px, py in pts:
        if not merged:
            merged.append((float(px), float(py)))
            continue
        mx, my = merged[-1]
        if abs(px - mx) <= x_eps and abs(py - my) <= y_eps:
            merged[-1] = ((mx + px) / 2.0, (my + py) / 2.0)
        else:
            merged.append((float(px), float(py)))
    return merged


def _build_human_prior_for_image(w: int, h: int) -> tuple[list[tuple[float, float]], float, float]:
    raw = _merge_near_points(HUMAN_ANNOTATED_CENTERS_RAW, x_eps=10.0, y_eps=10.0)
    sx = float(w / REF_STEP5_WIDTH)
    sy = float(h / REF_STEP5_HEIGHT)

    scaled = [(px * sx, py * sy) for px, py in raw]
    scaled = sorted(scaled, key=lambda p: p[0])
    if len(scaled) <= 1:
        pitch = max(1.0, 3.7 * HUMAN_ANNOTATED_RADIUS_PX * sx)
    else:
        gaps = np.diff(np.array([p[0] for p in scaled], dtype=np.float32))
        pitch = float(np.median(gaps))

    r_scaled = float(HUMAN_ANNOTATED_RADIUS_PX * sx)
    return scaled, r_scaled, pitch


def _circle_color_features(
    bgr: np.ndarray,
    hsv: np.ndarray,
    cx: float,
    cy: float,
    r: float,
) -> tuple[float, float, float, float, float, float] | None:
    h, w = bgr.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    inner = d <= 0.68 * r
    ring = (d >= 1.02 * r) & (d <= 1.34 * r)
    if int(inner.sum()) < 80 or int(ring.sum()) < 80:
        return None

    h_ch, s_ch, v_ch = cv2.split(hsv)
    b_ch, g_ch, r_ch = cv2.split(bgr)

    h_in = float(np.median(h_ch[inner]))
    s_in = float(np.median(s_ch[inner]))
    v_in = float(np.median(v_ch[inner]))

    s_ring = float(np.median(s_ch[ring]))
    v_ring = float(np.median(v_ch[ring]))

    b_in = float(np.median(b_ch[inner]))
    g_in = float(np.median(g_ch[inner]))
    r_in = float(np.median(r_ch[inner]))

    blue_dom = float(b_in - (g_in + r_in) / 2.0)
    sat_contrast = float(s_in - s_ring)
    val_contrast = float(abs(v_in - v_ring))

    return h_in, s_in, v_in, blue_dom, sat_contrast, val_contrast


def _collect_top_circle_candidates(
    rectified_bgr: np.ndarray,
    search_h: int,
    r_target: float,
    r_min: int,
    r_max: int,
) -> list[CircleCandidate]:
    h, w = rectified_bgr.shape[:2]
    gray = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2HSV)
    top_blur = cv2.GaussianBlur(gray[:search_h, :], (9, 9), 0)

    candidates: list[CircleCandidate] = []
    for param2 in (18, 16, 14, 12, 10, 9):
        circles = cv2.HoughCircles(
            top_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(24, int(r_target * 1.5)),
            param1=100,
            param2=param2,
            minRadius=r_min,
            maxRadius=r_max,
        )
        if circles is None:
            continue

        for c in circles[0]:
            cx, cy, r = map(float, c)
            if not (0.08 * h <= cy <= 0.48 * h):
                continue

            feats = _circle_color_features(rectified_bgr, hsv, cx, cy, r)
            if feats is None:
                continue
            h_in, s_in, v_in, blue_dom, sat_contrast, val_contrast = feats

            x_score = 1.0 - min(abs(cx - 0.5 * w) / (0.5 * w + 1.0), 1.0)
            y_score = 1.0 - min(abs(cy - 0.25 * h) / (0.34 * h + 1.0), 1.0)
            r_score = 1.0 - min(abs(r - r_target) / (r_max - r_min + 1.0), 1.0)
            geom_raw = 1.15 * y_score + 0.90 * r_score + 0.08 * x_score
            geom = float(np.clip(geom_raw / 2.13, 0.0, 1.0))

            # Initial broad color prior for blue reaction zone.
            hue_score = 1.0 - min(abs(h_in - 98.0) / 26.0, 1.0)
            sat_score = 1.0 - min(abs(s_in - 56.0) / 45.0, 1.0)
            blue_score = 1.0 - min(abs(blue_dom - 14.0) / 26.0, 1.0)
            sat_con_score = float(np.clip((sat_contrast + 8.0) / 40.0, 0.0, 1.0))
            val_con_score = float(np.clip(val_contrast / 80.0, 0.0, 1.0))
            color_pre = 0.35 * hue_score + 0.25 * sat_score + 0.25 * blue_score + 0.15 * (0.5 * sat_con_score + 0.5 * val_con_score)

            score = 0.67 * geom + 0.33 * color_pre + (18 - param2) * 0.012
            conf = float(np.clip(0.18 + 0.46 * geom + 0.36 * color_pre, 0.0, 1.0))

            candidates.append(
                CircleCandidate(
                    cx=cx,
                    cy=cy,
                    r=r,
                    geom_score=float(geom),
                    pre_color_score=float(color_pre),
                    ref_color_score=float(color_pre),
                    score=float(score),
                    confidence=conf,
                    method=f"hough_p{param2}",
                    h=h_in,
                    s=s_in,
                    v=v_in,
                    blue_dom=blue_dom,
                    sat_contrast=sat_contrast,
                    val_contrast=val_contrast,
                )
            )

    return candidates


def _dedupe_candidates(candidates: list[CircleCandidate]) -> list[CircleCandidate]:
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
    kept: list[CircleCandidate] = []
    for cand in ordered:
        duplicate = False
        for k in kept:
            dist = float(np.hypot(cand.cx - k.cx, cand.cy - k.cy))
            if dist < 0.46 * (cand.r + k.r):
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)

    return sorted(kept, key=lambda c: c.cx)


def _select_reference(
    candidates: list[CircleCandidate],
    w: int,
    h: int,
    r_target: float,
) -> tuple[CircleCandidate, CircleCandidate | None, float, float, float, float, float, float, float]:
    scores = np.array([c.score for c in candidates], dtype=np.float32)
    q_thr = float(np.percentile(scores, 50)) if len(scores) else -1e9
    quality_pool = [c for c in candidates if c.score >= q_thr]
    if not quality_pool:
        quality_pool = candidates

    # Anchor-1: mimic ROI01 behavior -> use the left-most high-quality circle.
    left_pool = [c for c in candidates if c.cx < 0.45 * w and c.r >= 0.72 * r_target and c.cy < 0.45 * h]
    if not left_pool:
        left_pool = [c for c in candidates if c.cx < 0.45 * w]
    seed_left = min(left_pool, key=lambda c: c.cx) if left_pool else min(candidates, key=lambda c: c.cx)

    # Anchor-2: ROI06-like position around center-right.
    mid_pool = [c for c in quality_pool if 0.45 * w <= c.cx <= 0.70 * w and c.cx > seed_left.cx + 1.6 * r_target]
    seed_mid = max(mid_pool, key=lambda c: c.score) if mid_pool else None
    if seed_mid is None:
        right_pool = [c for c in quality_pool if c.cx > seed_left.cx + 1.8 * r_target]
        if right_pool:
            seed_mid = max(right_pool, key=lambda c: c.score - 0.003 * abs(c.cx - 0.56 * w))

    y_ref = float(np.median([seed_left.cy, seed_mid.cy if seed_mid else 0.25 * h]))
    r_ref = float(np.median([seed_left.r, seed_mid.r if seed_mid else r_target, r_target]))

    if seed_mid is not None and seed_mid.cx > seed_left.cx:
        dx_est = float((seed_mid.cx - seed_left.cx) / 5.0)
    else:
        dx_est = float(2.8 * r_ref)

    # Calibrate dx with observed neighbor gaps from high-quality candidates.
    x_pool = sorted([c.cx for c in quality_pool if c.r >= 0.68 * r_target and c.cy < 0.48 * h])
    if len(x_pool) >= 3:
        gaps = np.diff(np.array(x_pool, dtype=np.float32))
        valid_gaps = gaps[(gaps >= 1.2 * r_target) & (gaps <= 5.2 * r_target)]
        if len(valid_gaps) > 0:
            gap_med = float(np.median(valid_gaps))
            dx_est = 0.55 * dx_est + 0.45 * gap_med
    dx_est = float(np.clip(dx_est, 1.55 * r_ref, 3.9 * r_ref))

    if seed_mid is not None:
        h_ref = float((seed_left.h + seed_mid.h) / 2.0)
        s_ref = float((seed_left.s + seed_mid.s) / 2.0)
        v_ref = float((seed_left.v + seed_mid.v) / 2.0)
        b_ref = float((seed_left.blue_dom + seed_mid.blue_dom) / 2.0)
    else:
        h_ref, s_ref, v_ref, b_ref = seed_left.h, seed_left.s, seed_left.v, seed_left.blue_dom

    return seed_left, seed_mid, y_ref, r_ref, dx_est, h_ref, s_ref, v_ref, b_ref


def _rescore_with_reference(
    candidates: list[CircleCandidate],
    h_ref: float,
    s_ref: float,
    v_ref: float,
    b_ref: float,
) -> list[CircleCandidate]:
    rescored: list[CircleCandidate] = []
    for c in candidates:
        h_score = 1.0 - min(abs(c.h - h_ref) / 18.0, 1.0)
        s_score = 1.0 - min(abs(c.s - s_ref) / 35.0, 1.0)
        v_score = 1.0 - min(abs(c.v - v_ref) / 65.0, 1.0)
        b_score = 1.0 - min(abs(c.blue_dom - b_ref) / 22.0, 1.0)

        sat_con_score = float(np.clip((c.sat_contrast + 10.0) / 45.0, 0.0, 1.0))
        val_con_score = float(np.clip(c.val_contrast / 85.0, 0.0, 1.0))
        color_ref = 0.32 * h_score + 0.20 * s_score + 0.16 * v_score + 0.22 * b_score + 0.10 * (0.5 * sat_con_score + 0.5 * val_con_score)

        new_score = 0.58 * c.geom_score + 0.42 * color_ref
        new_conf = float(np.clip(0.22 + 0.40 * c.geom_score + 0.38 * color_ref, 0.0, 1.0))

        rescored.append(
            CircleCandidate(
                cx=c.cx,
                cy=c.cy,
                r=c.r,
                geom_score=c.geom_score,
                pre_color_score=c.pre_color_score,
                ref_color_score=float(color_ref),
                score=float(new_score),
                confidence=new_conf,
                method=c.method,
                h=c.h,
                s=c.s,
                v=c.v,
                blue_dom=c.blue_dom,
                sat_contrast=c.sat_contrast,
                val_contrast=c.val_contrast,
            )
        )

    return sorted(rescored, key=lambda c: c.cx)


def _pick_candidate_for_expected(
    candidates: list[CircleCandidate],
    used_indices: set[int],
    ex: float,
    ey: float,
    expected_r: float,
    expected_pitch: float,
) -> int | None:
    wx = max(0.55 * expected_pitch, 1.35 * expected_r)
    wy = max(0.45 * expected_r, 24.0)

    best_idx: int | None = None
    best_score = -1e9
    for i, c in enumerate(candidates):
        if i in used_indices:
            continue
        if abs(c.cx - ex) > wx or abs(c.cy - ey) > wy:
            continue
        if c.r < 0.55 * expected_r or c.r > 1.75 * expected_r:
            continue

        rank = (
            1.20 * c.score
            + 0.25 * c.ref_color_score
            - 0.0060 * abs(c.cx - ex)
            - 0.0120 * abs(c.cy - ey)
            - 0.0080 * abs(c.r - expected_r)
        )
        if rank > best_score:
            best_score = rank
            best_idx = i

    return best_idx


def _local_hough_near_expected(
    rect_gray: np.ndarray,
    ex: float,
    ey: float,
    expected_r: float,
    expected_pitch: float,
) -> CircleDetection | None:
    h, w = rect_gray.shape[:2]
    wx = max(0.85 * expected_pitch, 1.55 * expected_r)
    wy = max(1.15 * expected_r, 42.0)

    x0 = clamp(int(round(ex - wx)), 0, max(0, w - 2))
    x1 = clamp(int(round(ex + wx)), x0 + 2, w)
    y0 = clamp(int(round(ey - wy)), 0, max(0, h - 2))
    y1 = clamp(int(round(ey + wy)), y0 + 2, h)

    patch = rect_gray[y0:y1, x0:x1]
    if patch.size == 0:
        return None

    blur = cv2.GaussianBlur(patch, (7, 7), 0)
    r_min = max(12, int(round(0.65 * expected_r)))
    r_max = max(r_min + 6, int(round(1.38 * expected_r)))

    best = None
    for param2 in (14, 12, 10):
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(16, int(round(0.85 * expected_r))),
            param1=100,
            param2=param2,
            minRadius=r_min,
            maxRadius=r_max,
        )
        if circles is None:
            continue
        for c in circles[0]:
            lx, ly, rr = map(float, c)
            gx = x0 + lx
            gy = y0 + ly
            # Strictly local nearest expected target.
            rank = (
                -0.0100 * abs(gx - ex)
                -0.0150 * abs(gy - ey)
                -0.0120 * abs(rr - expected_r)
                + (14 - param2) * 0.01
            )
            if best is None or rank > best[0]:
                best = (rank, gx, gy, rr, param2)

    if best is None:
        return None

    _, gx, gy, rr, p2 = best
    dist_x = abs(gx - ex) / (max(1.0, 0.75 * expected_pitch))
    dist_y = abs(gy - ey) / (max(1.0, 0.60 * expected_r))
    dist_r = abs(rr - expected_r) / (max(1.0, 0.60 * expected_r))
    conf = float(np.clip(0.30 + 0.45 * (1.0 - min(1.0, 0.5 * (dist_x + dist_y))) + 0.25 * (1.0 - min(1.0, dist_r)), 0.0, 1.0))
    return CircleDetection(cx=float(gx), cy=float(gy), r=float(rr), confidence=conf, method=f"human_local_hough_p{p2}")


def _match_with_human_prior(
    rectified_bgr: np.ndarray,
    candidates: list[CircleCandidate],
    expected_centers: list[tuple[float, float]],
    expected_r: float,
    expected_pitch: float,
) -> tuple[list[CircleDetection], int]:
    rect_gray = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2GRAY)
    used: set[int] = set()
    detections: list[CircleDetection] = []
    fallback_count = 0

    for ex, ey in expected_centers:
        idx = _pick_candidate_for_expected(
            candidates,
            used_indices=used,
            ex=ex,
            ey=ey,
            expected_r=expected_r,
            expected_pitch=expected_pitch,
        )
        if idx is not None:
            used.add(idx)
            c = candidates[idx]
            pos_term = 1.0 - min(1.0, abs(c.cx - ex) / (0.65 * expected_pitch + 1.0))
            conf = float(np.clip(0.55 * c.confidence + 0.45 * pos_term, 0.0, 1.0))
            # Pull detection toward human annotation to honor manual labels.
            bx = 0.62 * ex + 0.38 * c.cx
            by = 0.62 * ey + 0.38 * c.cy
            br = 0.45 * expected_r + 0.55 * c.r
            detections.append(
                CircleDetection(
                    cx=float(bx),
                    cy=float(by),
                    r=float(br),
                    confidence=conf,
                    method=f"{c.method}_humanprior",
                )
            )
            continue

        local = _local_hough_near_expected(
            rect_gray,
            ex=ex,
            ey=ey,
            expected_r=expected_r,
            expected_pitch=expected_pitch,
        )
        if local is not None:
            # Local result is also centered by annotation prior.
            local.cx = float(0.55 * ex + 0.45 * local.cx)
            local.cy = float(0.55 * ey + 0.45 * local.cy)
            local.r = float(0.50 * expected_r + 0.50 * local.r)
            detections.append(local)
            fallback_count += 1
            continue

        detections.append(
            CircleDetection(
                cx=float(ex),
                cy=float(ey),
                r=float(expected_r),
                confidence=0.68,
                method="human_prior_geom",
            )
        )
        fallback_count += 1

    return sorted(detections, key=lambda d: d.cx), fallback_count


def _candidate_rank(cand: CircleCandidate, expected_x: float, y_ref: float, r_ref: float) -> float:
    return (
        cand.score
        - 0.0052 * abs(cand.cx - expected_x)
        - 0.0080 * abs(cand.cy - y_ref)
        - 0.0100 * abs(cand.r - r_ref)
    )


def _pick_candidate_near_expected(
    candidates: list[CircleCandidate],
    used_indices: set[int],
    expected_x: float,
    current_x: float,
    dx_est: float,
    y_ref: float,
    r_ref: float,
    direction: int,
) -> int | None:
    x_lo = expected_x - 0.60 * dx_est
    x_hi = expected_x + 0.60 * dx_est

    best_idx: int | None = None
    best_score = -1e9
    for i, cand in enumerate(candidates):
        if i in used_indices:
            continue
        if not (x_lo <= cand.cx <= x_hi):
            continue
        if direction > 0 and cand.cx <= current_x + 0.12 * dx_est:
            continue
        if direction < 0 and cand.cx >= current_x - 0.12 * dx_est:
            continue

        score = _candidate_rank(cand, expected_x=expected_x, y_ref=y_ref, r_ref=r_ref)
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def _track_by_translation(
    candidates: list[CircleCandidate],
    seed_idx: int,
    dx_est: float,
    y_ref: float,
    r_ref: float,
    w: int,
) -> list[int]:
    used: set[int] = {seed_idx}
    right_path: list[int] = [seed_idx]

    cur_x = candidates[seed_idx].cx
    cur_y = candidates[seed_idx].cy
    miss = 0
    while cur_x + 0.75 * dx_est < w:
        expected_x = cur_x + dx_est
        idx = _pick_candidate_near_expected(
            candidates,
            used_indices=used,
            expected_x=expected_x,
            current_x=cur_x,
            dx_est=dx_est,
            y_ref=y_ref,
            r_ref=r_ref,
            direction=1,
        )
        if idx is None:
            miss += 1
            cur_x = expected_x
            if miss > 4:
                break
            continue

        used.add(idx)
        right_path.append(idx)
        cur_x = candidates[idx].cx
        cur_y = candidates[idx].cy
        y_ref = 0.88 * y_ref + 0.12 * cur_y
        miss = 0

    left_path: list[int] = []
    cur_x = candidates[seed_idx].cx
    miss = 0
    while cur_x - 0.75 * dx_est > 0:
        expected_x = cur_x - dx_est
        idx = _pick_candidate_near_expected(
            candidates,
            used_indices=used,
            expected_x=expected_x,
            current_x=cur_x,
            dx_est=dx_est,
            y_ref=y_ref,
            r_ref=r_ref,
            direction=-1,
        )
        if idx is None:
            miss += 1
            cur_x = expected_x
            if miss > 3:
                break
            continue

        used.add(idx)
        left_path.append(idx)
        cur_x = candidates[idx].cx
        miss = 0

    return list(reversed(left_path)) + right_path


def _supplement_untracked(
    candidates: list[CircleCandidate],
    selected_indices: list[int],
    y_ref: float,
    r_ref: float,
    dx_est: float,
    h: int,
) -> list[int]:
    selected_set = set(selected_indices)
    score_values = np.array([c.score for c in candidates], dtype=np.float32)
    quality_thr = float(np.percentile(score_values, 50)) if len(score_values) else -1e9

    for i, cand in enumerate(candidates):
        if i in selected_set:
            continue
        if cand.score < quality_thr:
            continue
        if cand.ref_color_score < 0.35:
            continue
        if abs(cand.cy - y_ref) > 0.23 * h:
            continue
        if cand.r < 0.72 * r_ref or cand.r > 1.35 * r_ref:
            continue

        too_close = False
        for j in selected_indices:
            if abs(cand.cx - candidates[j].cx) < 0.55 * dx_est:
                too_close = True
                break
        if too_close:
            continue

        selected_indices.append(i)
        selected_set.add(i)

    return sorted(selected_indices, key=lambda i: candidates[i].cx)


def _prune_spacing_outliers(candidates: list[CircleCandidate], selected_indices: list[int]) -> list[int]:
    if len(selected_indices) < 4:
        return sorted(selected_indices, key=lambda i: candidates[i].cx)

    selected = sorted(selected_indices, key=lambda i: candidates[i].cx)
    for _ in range(8):
        if len(selected) < 4:
            break
        xs = [candidates[i].cx for i in selected]
        gaps = np.diff(xs)
        if len(gaps) == 0:
            break
        med_gap = float(np.median(gaps))
        if med_gap < 1e-6:
            break
        bad = np.where(gaps < 0.58 * med_gap)[0]
        if len(bad) == 0:
            break
        k = int(bad[0])
        i1 = selected[k]
        i2 = selected[k + 1]
        drop = i1 if candidates[i1].score < candidates[i2].score else i2
        selected.remove(drop)

    return sorted(selected, key=lambda i: candidates[i].cx)


def _trim_to_expected_count(
    candidates: list[CircleCandidate],
    selected_indices: list[int],
    dx_est: float,
) -> list[int]:
    selected = sorted(selected_indices, key=lambda i: candidates[i].cx)
    if len(selected) < 3 or dx_est <= 1.0:
        return selected

    x_min = candidates[selected[0]].cx
    x_max = candidates[selected[-1]].cx
    expected_count = max(3, int(round((x_max - x_min) / dx_est)) + 1)
    if len(selected) <= expected_count:
        return selected

    while len(selected) > expected_count and len(selected) > 2:
        interior = selected[1:-1]
        if not interior:
            break

        scores = np.array([candidates[i].score for i in selected], dtype=np.float32)
        s_min = float(scores.min())
        s_max = float(scores.max())
        s_span = max(1e-6, s_max - s_min)

        best_drop = None
        best_cost = 1e9
        for drop in interior:
            trial = [i for i in selected if i != drop]
            xs = np.array([candidates[i].cx for i in trial], dtype=np.float32)
            if len(xs) < 3:
                continue
            gaps = np.diff(xs)
            mean_gap = float(np.mean(gaps))
            std_gap = float(np.std(gaps))
            cv = std_gap / (mean_gap + 1e-6)

            # Lower score points are easier to drop.
            score_norm = (candidates[drop].score - s_min) / s_span
            score_penalty = 0.18 * score_norm

            cost = cv + score_penalty
            if cost < best_cost:
                best_cost = cost
                best_drop = drop

        if best_drop is None:
            break
        selected.remove(best_drop)

    return sorted(selected, key=lambda i: candidates[i].cx)


def _fallback_equal_split(
    rect_gray: np.ndarray,
    count_hint: int,
    search_h: int,
    r_target: float,
    r_min: int,
    r_max: int,
) -> list[CircleDetection]:
    h, w = rect_gray.shape[:2]
    count_hint = max(1, count_hint)
    detections: list[CircleDetection] = []

    for i in range(count_hint):
        x0 = int(round(i * w / count_hint))
        x1 = int(round((i + 1) * w / count_hint))
        x0 = clamp(x0, 0, w - 1)
        x1 = clamp(x1, x0 + 1, w)

        lane = rect_gray[:search_h, x0:x1]
        blur = cv2.GaussianBlur(lane, (9, 9), 0)
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(20, int((x1 - x0) * 0.35)),
            param1=100,
            param2=12,
            minRadius=r_min,
            maxRadius=r_max,
        )
        if circles is not None:
            best = None
            for c in circles[0]:
                cx, cy, r = map(float, c)
                score = -abs(cx - (x1 - x0) / 2.0) - 0.8 * abs(cy - 0.25 * h)
                if best is None or score > best[0]:
                    best = (score, cx, cy, r)
            if best is not None:
                _, cx, cy, r = best
                detections.append(
                    CircleDetection(
                        cx=float(x0 + cx),
                        cy=float(cy),
                        r=float(r),
                        confidence=0.45,
                        method="fallback_split_hough",
                    )
                )
                continue

        detections.append(
            CircleDetection(
                cx=float((x0 + x1) / 2.0),
                cy=float(0.25 * h),
                r=float(r_target),
                confidence=0.0,
                method="fallback_split_geom",
            )
        )

    return detections


def run_step67(
    rectified_bgr: np.ndarray,
    num_strips: int,
    output_dir: Path,
    debug: bool = False,
) -> dict[str, Any]:
    """
    Step 6/7 optimized with geometry + HSV/RGB color fusion.
    Not hard-coded to 10 circles; num_strips is only a fallback hint.
    """
    h, w = rectified_bgr.shape[:2]
    rect_gray = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2GRAY)

    num_hint = int(num_strips)
    search_h = int(h * 0.62)

    # User-guided diameter prior: around 125 px on step_05 image.
    d_target = 125.0
    r_target = d_target / 2.0
    r_min = max(28, int(round(r_target * 0.72)))
    r_max = min(int(round(r_target * 1.40)), int(round(0.42 * h)))
    r_max = max(r_min + 8, r_max)

    raw_candidates = _collect_top_circle_candidates(
        rectified_bgr,
        search_h=search_h,
        r_target=r_target,
        r_min=r_min,
        r_max=r_max,
    )
    candidates = _dedupe_candidates(raw_candidates)

    selected_detections: list[CircleDetection] = []
    seed_left = None
    seed_mid = None
    dx_est = 0.0
    y_ref = 0.25 * h
    r_ref = r_target
    h_ref = s_ref = v_ref = b_ref = None
    human_expected_centers, human_expected_r, human_expected_pitch = _build_human_prior_for_image(w=w, h=h)
    human_prior_fallbacks = 0

    if candidates:
        (
            seed_left,
            seed_mid,
            y_ref,
            r_ref,
            dx_est,
            h_ref,
            s_ref,
            v_ref,
            b_ref,
        ) = _select_reference(candidates, w=w, h=h, r_target=r_target)

        candidates = _rescore_with_reference(candidates, h_ref=h_ref, s_ref=s_ref, v_ref=v_ref, b_ref=b_ref)

        # First choice: human-prior matching (from user labels), not fixed to 10.
        if len(human_expected_centers) >= 3:
            selected_detections, human_prior_fallbacks = _match_with_human_prior(
                rectified_bgr=rectified_bgr,
                candidates=candidates,
                expected_centers=human_expected_centers,
                expected_r=human_expected_r,
                expected_pitch=human_expected_pitch,
            )
            y_ref = float(np.median([d.cy for d in selected_detections])) if selected_detections else y_ref
            r_ref = float(np.median([d.r for d in selected_detections])) if selected_detections else r_ref
            dx_est = float(human_expected_pitch)

        # Fallback: geometry/color translation tracker when no prior is available.
        if not selected_detections:
            seed_idx = min(
                range(len(candidates)),
                key=lambda i: abs(candidates[i].cx - seed_left.cx) + abs(candidates[i].cy - seed_left.cy),
            )

            selected_indices = _track_by_translation(
                candidates,
                seed_idx=seed_idx,
                dx_est=dx_est,
                y_ref=y_ref,
                r_ref=r_ref,
                w=w,
            )
            selected_indices = _supplement_untracked(
                candidates,
                selected_indices=selected_indices,
                y_ref=y_ref,
                r_ref=r_ref,
                dx_est=dx_est,
                h=h,
            )
            selected_indices = _prune_spacing_outliers(candidates, selected_indices)

            selected_detections = [
                CircleDetection(
                    cx=float(candidates[i].cx),
                    cy=float(candidates[i].cy),
                    r=float(candidates[i].r),
                    confidence=float(candidates[i].confidence),
                    method=str(candidates[i].method),
                )
                for i in selected_indices
                if candidates[i].score >= 0.48
            ]

    if not selected_detections:
        fallback_count = num_hint if num_hint > 0 else 10
        selected_detections = _fallback_equal_split(
            rect_gray,
            count_hint=fallback_count,
            search_h=search_h,
            r_target=r_target,
            r_min=r_min,
            r_max=r_max,
        )
        dx_est = float(w / max(1, len(selected_detections)))
        y_ref = 0.25 * h
        r_ref = r_target

    selected_detections = sorted(selected_detections, key=lambda d: d.cx)

    # Dynamic boundaries from neighboring centers.
    n = len(selected_detections)
    centers = [d.cx for d in selected_detections]
    boundaries: list[tuple[int, int]] = []
    for i, cx in enumerate(centers):
        if n == 1:
            left = int(max(0, round(cx - 1.3 * r_ref)))
            right = int(min(w, round(cx + 1.3 * r_ref)))
        else:
            if i == 0:
                left = int(max(0, round(cx - 0.5 * (centers[1] - cx))))
            else:
                left = int(round((centers[i - 1] + cx) / 2.0))
            if i == n - 1:
                right = int(min(w, round(cx + 0.5 * (cx - centers[i - 1]))))
            else:
                right = int(round((cx + centers[i + 1]) / 2.0))
        left = clamp(left, 0, w - 1)
        right = clamp(right, left + 1, w)
        boundaries.append((left, right))

    lane_vis = rectified_bgr.copy()
    roi_paths: list[Path] = []
    detections_payload: list[dict[str, Any]] = []
    warnings: list[str] = []

    for i, detection in enumerate(selected_detections, start=1):
        left, right = boundaries[i - 1]
        lane_w = right - left

        # Radius stabilization toward user target diameter (~125px).
        r_use = 0.65 * detection.r + 0.35 * r_target
        r_use = float(np.clip(r_use, 0.80 * r_target, 1.20 * r_target))
        detection.r = r_use

        roi, mask_ratio = make_roi(rectified_bgr, detection.cx, detection.cy, detection.r, size_scale=2.20)
        roi_file = output_dir / f"roi_{i:02d}.png"
        write_image(roi_file, roi)
        roi_paths.append(roi_file)

        center_offset_norm = float(abs(detection.cx - (left + right) / 2.0) / (lane_w / 2.0 + 1.0))
        lane_warning = ""
        if detection.confidence < 0.55:
            lane_warning = "low_confidence"
        if center_offset_norm > 0.50:
            lane_warning = f"{lane_warning}|center_offset_high" if lane_warning else "center_offset_high"
        if mask_ratio < 0.30 or mask_ratio > 0.80:
            lane_warning = f"{lane_warning}|mask_ratio_out_of_range" if lane_warning else "mask_ratio_out_of_range"
        if lane_warning:
            warnings.append(f"ROI {i}: {lane_warning}")

        cv2.line(lane_vis, (left, 0), (left, h - 1), (120, 120, 120), 1)
        if i == n:
            cv2.line(lane_vis, (right - 1, 0), (right - 1, h - 1), (120, 120, 120), 1)

        color = (0, 255, 0)
        if "fallback" in detection.method:
            color = (0, 0, 255)
        cv2.circle(
            lane_vis,
            (int(round(detection.cx)), int(round(detection.cy))),
            int(round(detection.r)),
            color,
            2,
        )
        cv2.putText(
            lane_vis,
            f"{i}",
            (int(max(0, detection.cx - 8)), int(max(18, detection.cy - detection.r - 6))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        detections_payload.append(
            {
                "lane_id": i,
                "lane_bbox_rectified_xyxy": [left, 0, right, h],
                "center_rectified_xy": [round(float(detection.cx), 2), round(float(detection.cy), 2)],
                "radius_px": round(float(detection.r), 2),
                "method": detection.method,
                "confidence": round(float(detection.confidence), 4),
                "mask_ratio": round(mask_ratio, 4),
                "center_offset_norm": round(center_offset_norm, 4),
                "warning": lane_warning,
                "roi_file": roi_file.name,
            }
        )

    write_image(output_dir / "step_06_lanes_and_circles.png", lane_vis)

    if debug:
        seed_left_info = None if seed_left is None else [round(seed_left.cx, 2), round(seed_left.cy, 2), round(seed_left.r, 2)]
        seed_mid_info = None if seed_mid is None else [round(seed_mid.cx, 2), round(seed_mid.cy, 2), round(seed_mid.r, 2)]
        color_ref = None if h_ref is None else [round(float(h_ref), 2), round(float(s_ref), 2), round(float(v_ref), 2), round(float(b_ref), 2)]
        hp_preview = [[round(x, 2), round(y, 2)] for x, y in human_expected_centers[:5]]
        print(
            "[step67] "
            f"h={h}, w={w}, hint={num_hint}, candidates_raw={len(raw_candidates)}, candidates_dedup={len(candidates)}, "
            f"detected={len(selected_detections)}, search_h={search_h}, radius_range=[{r_min}, {r_max}], "
            f"seed_left={seed_left_info}, seed_mid={seed_mid_info}, color_ref={color_ref}, "
            f"human_prior_count={len(human_expected_centers)}, human_prior_preview={hp_preview}, "
            f"human_prior_fallbacks={human_prior_fallbacks}, dx_est={round(dx_est, 2)}, warnings={len(warnings)}"
        )

    return {
        "num_rois_saved": len(roi_paths),
        "detected_without_fallback": int(sum(1 for d in detections_payload if "fallback" not in d["method"])),
        "detections": detections_payload,
        "warnings": warnings,
        "algorithm": {
            "name": "seed_translation_hsv_rgb_fusion",
            "num_hint": num_hint,
            "search_h": search_h,
            "diameter_target_px": d_target,
            "radius_range": [r_min, r_max],
            "seed_left": None if seed_left is None else [round(seed_left.cx, 2), round(seed_left.cy, 2), round(seed_left.r, 2)],
            "seed_mid": None if seed_mid is None else [round(seed_mid.cx, 2), round(seed_mid.cy, 2), round(seed_mid.r, 2)],
            "color_reference_hsv_blue": None if h_ref is None else [round(float(h_ref), 3), round(float(s_ref), 3), round(float(v_ref), 3), round(float(b_ref), 3)],
            "dx_est": round(float(dx_est), 3),
            "reference_y": round(float(y_ref), 3),
            "reference_r": round(float(r_ref), 3),
            "candidate_count_raw": int(len(raw_candidates)),
            "candidate_count_dedup": int(len(candidates)),
            "human_prior_enabled": True,
            "human_prior_count": int(len(human_expected_centers)),
            "human_prior_radius_px": round(float(human_expected_r), 3),
            "human_prior_pitch_px": round(float(human_expected_pitch), 3),
            "human_prior_fallback_count": int(human_prior_fallbacks),
        },
    }
