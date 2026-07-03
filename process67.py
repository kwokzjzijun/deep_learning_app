#!/usr/bin/env python3
"""
Non-YOLO demo for extracting top circular reaction ROIs from one strip image.

Usage:
  python3 test.py --input /Users/gzj/Desktop/ua_demo/IMG_2.PNG --debug
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from step67 import run_step67


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract top circular reaction ROIs from one strip image (non-YOLO demo)."
    )
    parser.add_argument(
        "--input",
        default="/Users/gzj/Desktop/ua_demo/IMG_2.PNG",
        help="Input image path.",
    )
    parser.add_argument(
        "--output",
        default="/Users/gzj/Desktop/ua_demo/output_demo",
        help="Output directory.",
    )
    parser.add_argument(
        "--num-strips",
        type=int,
        default=0,
        help="Optional count hint for strip pieces. Use 0 for fully auto circle count.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug logs and save extra visualizations.",
    )
    return parser.parse_args()


def log(message: str, debug: bool) -> None:
    if debug:
        print(message)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def choose_strip_component(gray: np.ndarray, debug: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """Find the long strip component using dark-threshold masks + component scoring."""
    h, w = gray.shape[:2]
    image_area = float(h * w)

    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask_variants: list[tuple[str, np.ndarray]] = [
        ("otsu_inv", otsu_inv),
        ("fixed_130", (gray < 130).astype(np.uint8) * 255),
        ("fixed_120", (gray < 120).astype(np.uint8) * 255),
        ("fixed_110", (gray < 110).astype(np.uint8) * 255),
    ]

    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
    ]

    candidates: list[tuple[float, np.ndarray, dict[str, Any]]] = []
    for mask_name, base_mask in mask_variants:
        for k_idx, kernel in enumerate(kernels):
            proc = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, kernel)
            proc = cv2.morphologyEx(proc, cv2.MORPH_OPEN, kernel)
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(proc, connectivity=8)
            for i in range(1, n_labels):
                area = float(stats[i, cv2.CC_STAT_AREA])
                if area < image_area * 0.0015 or area > image_area * 0.6:
                    continue
                comp = (labels == i).astype(np.uint8) * 255
                contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                contour = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(contour)
                rw, rh = rect[1]
                if rw < 2 or rh < 2:
                    continue
                aspect = max(rw, rh) / max(1.0, min(rw, rh))
                if aspect < 2.8:
                    continue
                # Prefer larger, elongated components.
                score = area * (1.0 + min(aspect, 20.0) / 12.0)
                info = {
                    "mask_name": mask_name,
                    "kernel_index": k_idx,
                    "area": area,
                    "aspect": aspect,
                }
                candidates.append((score, comp, info))

    if not candidates:
        log("[WARN] No candidate passed strict filters, fallback to largest fixed_120 component.", debug)
        fallback_mask = (gray < 120).astype(np.uint8) * 255
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fallback_mask, connectivity=8)
        if n_labels <= 1:
            raise RuntimeError("Cannot locate strip component in the image.")
        largest_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        comp = (labels == largest_idx).astype(np.uint8) * 255
        fallback_info = {
            "mask_name": "fixed_120_fallback",
            "kernel_index": -1,
            "area": float(stats[largest_idx, cv2.CC_STAT_AREA]),
            "aspect": -1.0,
        }
        return comp, fallback_info

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_comp, best_info = candidates[0]
    best_info["score"] = best_score
    return best_comp, best_info


def rectified_strip_view(
    image_bgr: np.ndarray, strip_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return rectified strip image and transform matrices."""
    contours, _ = cv2.findContours(strip_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contour in strip mask.")
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    ordered = order_points(box)

    # Slightly expand to avoid clipping strip edges.
    center = ordered.mean(axis=0, keepdims=True)
    ordered = center + (ordered - center) * 1.03

    width_a = np.linalg.norm(ordered[2] - ordered[3])
    width_b = np.linalg.norm(ordered[1] - ordered[0])
    height_a = np.linalg.norm(ordered[1] - ordered[2])
    height_b = np.linalg.norm(ordered[0] - ordered[3])
    max_w = max(10, int(round(max(width_a, width_b))))
    max_h = max(10, int(round(max(height_a, height_b))))

    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype=np.float32
    )
    mat = cv2.getPerspectiveTransform(ordered, dst)
    inv_mat = cv2.getPerspectiveTransform(dst, ordered)
    warped = cv2.warpPerspective(image_bgr, mat, (max_w, max_h))

    # Normalize orientation: long direction horizontal.
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped, ordered, inv_mat


def write_image(path: Path, image: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def cleanup_output_images(output_dir: Path) -> None:
    # Keep output deterministic per run: remove stale step/roi PNG files first.
    for pattern in ("step_*.png", "roi_*.png"):
        for p in output_dir.glob(pattern):
            if p.is_file():
                p.unlink()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_output_images(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    image_bgr = cv2.imread(str(input_path))
    if image_bgr is None:
        raise RuntimeError(f"Cannot read image: {input_path}")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 1: input image.
    write_image(output_dir / "step_01_input.png", image_bgr)

    # Step 2: strip localization.
    strip_mask, strip_info = choose_strip_component(gray_blur, debug=args.debug)

    strip_contours, _ = cv2.findContours(strip_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not strip_contours:
        raise RuntimeError("No strip contour found after strip localization.")
    strip_contour = max(strip_contours, key=cv2.contourArea)
    strip_box = cv2.boxPoints(cv2.minAreaRect(strip_contour)).astype(np.int32)
    strip_box_vis = image_bgr.copy()
    cv2.drawContours(strip_box_vis, [strip_box], -1, (0, 255, 0), 8)
    write_image(output_dir / "step_04_strip_box.png", strip_box_vis)

    # Step 3: perspective rectification.
    rectified_bgr, ordered_box, _ = rectified_strip_view(image_bgr, strip_mask)

    # Step 6 + 7: top-circle detection + ROI extraction (modularized).
    num_strips = int(args.num_strips)
    step67_result = run_step67(
        rectified_bgr=rectified_bgr,
        num_strips=num_strips,
        output_dir=output_dir,
        debug=args.debug,
    )
    detections_payload = step67_result["detections"]
    warnings = step67_result["warnings"]

    results = {
        "input_image": str(input_path),
        "output_dir": str(output_dir),
        "num_strips_hint": num_strips,
        "num_strips_expected": num_strips,
        "num_rois_saved": int(step67_result["num_rois_saved"]),
        "detected_without_fallback": int(step67_result["detected_without_fallback"]),
        "strip_localization": {
            "selected_mask": strip_info.get("mask_name"),
            "selected_kernel_index": strip_info.get("kernel_index"),
            "component_area_px": round(float(strip_info.get("area", 0.0)), 2),
            "component_aspect": round(float(strip_info.get("aspect", 0.0)), 4),
            "ordered_box_points_xy": [[round(float(x), 2), round(float(y), 2)] for x, y in ordered_box],
        },
        "step67_algorithm": step67_result.get("algorithm", {}),
        "detections": detections_payload,
        "warnings": warnings,
        "annotation_recommendation": {
            "yolo_detection_classes": ["reaction_roi"],
            "optional_extra_class": "strip",
            "do_not_use_concentration_as_detection_class": True,
            "label_binding": {
                "single_concentration_per_image": "inherit concentration from filename/csv metadata to each cropped ROI",
                "mixed_concentration_per_image": "use external mapping table: image_id + lane_id -> concentration",
            },
        },
    }

    results_path = output_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved outputs to: {output_dir}")
    print(f"Saved results JSON: {results_path}")
    print(f"ROIs saved: {step67_result['num_rois_saved']} (num-strips hint {num_strips})")
    if warnings:
        print(f"Warnings: {len(warnings)}")
    else:
        print("Warnings: 0")


if __name__ == "__main__":
    main()
