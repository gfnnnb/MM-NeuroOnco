import os
import json
import cv2
import math
import argparse
import numpy as np
from tqdm import tqdm

# --- Thresholds ---
TH_SIZE_SMALL = 0.01  # < 1%
TH_SIZE_LARGE = 0.05  # >= 5%

TH_CIRCULARITY_IRREGULAR = 0.5
TH_CIRCULARITY_ROUND = 0.8
TH_ELONGATION_ROUND = 1.5

TH_SPREAD_FCORE = 0.7


def analyze_size(mask_area, img_h, img_w):
    total_pixels = img_h * img_w
    if total_pixels == 0:
        return "Unknown"
    r_area = mask_area / total_pixels
    if r_area < TH_SIZE_SMALL:
        return "Small/Focal"
    elif r_area < TH_SIZE_LARGE:
        return "Medium"
    else:
        return "Large/Extensive"


def analyze_shape(contour, area):
    if contour is None or area == 0:
        return "Unknown"
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    elongation = 1.0
    if len(contour) >= 5:
        try:
            (_, _), (MA, ma), _ = cv2.fitEllipse(contour)
            major = max(MA, ma)
            minor = min(MA, ma)
            if minor > 0:
                elongation = major / minor
        except Exception:
            pass

    if circularity < TH_CIRCULARITY_IRREGULAR:
        return "Irregular/Spiculated"
    elif circularity >= TH_CIRCULARITY_ROUND and elongation <= TH_ELONGATION_ROUND:
        return "Round/Oval"
    else:
        return "Lobulated"


def analyze_spread(stats, num_labels):
    n_c = num_labels - 1
    if n_c == 0:
        return "None"
    areas = stats[1:, cv2.CC_STAT_AREA]
    a_max = np.max(areas)
    total_area = np.sum(areas)
    f_core = a_max / total_area if total_area > 0 else 0

    if n_c == 1:
        return "Solitary Lesion"
    return "Dominant Core with Satellites" if f_core >= TH_SPREAD_FCORE else "Scattered/Multifocal"


def analyze_location(mask_binary, img_h, img_w):
    moments = cv2.moments(mask_binary, binaryImage=True)
    m00 = moments.get("m00", 0)
    if m00 == 0:
        return "Unknown"
    cx = moments["m10"] / m00
    cy = moments["m01"] / m00

    w_third = img_w / 3
    if cx < w_third:
        h_pos = "Left"
    elif cx < 2 * w_third:
        h_pos = "Center"
    else:
        h_pos = "Right"

    h_third = img_h / 3
    if cy < h_third:
        v_pos = "Upper"
    elif cy < 2 * h_third:
        v_pos = "Center"
    else:
        v_pos = "Lower"

    return "Center of the Image" if (v_pos == "Center" and h_pos == "Center") else f"{v_pos}-{h_pos} Region"


def safe_join(root: str, rel_path: str) -> str:
    """
    Join root + rel_path safely (avoid path traversal).
    Assumes rel_path should be relative (like "tumor/xx/mask.png").
    """
    rel_path = os.path.normpath(rel_path).lstrip(os.sep).lstrip("/")
    full = os.path.normpath(os.path.join(root, rel_path))
    root_norm = os.path.normpath(root)
    if not full.startswith(root_norm):
        raise ValueError(f"Unsafe path detected: {rel_path}")
    return full


def process_medical_semantics(dataset_root: str, input_json: str, output_json: str, drop_origin_json: bool = True, verbose: bool = False):
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Input file not found: {input_json}")

    print("🚀 Computing medical semantics and enriching metadata...")
    print(f"📥 input:  {input_json}")
    print(f"📁 root:   {dataset_root}")
    print(f"📤 output: {output_json}")

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of entries.")

    processed_count = 0
    null_count = 0

    for idx, entry in enumerate(tqdm(data, desc="Processing Semantics")):
        if drop_origin_json:
            entry.pop("origin_json", None)

        semantic_data = None

        try:
            if entry.get("has_mask") and entry.get("mask_path"):
                mask_path = entry["mask_path"]

                full_mask_path = safe_join(dataset_root, mask_path)
                if os.path.exists(full_mask_path):
                    mask = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        h, w = mask.shape
                        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

                        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

                        if num_labels > 1:
                            areas = stats[1:, cv2.CC_STAT_AREA]
                            total_tumor_area = int(np.sum(areas))

                            spread_label = analyze_spread(stats, num_labels)

                            max_idx = int(np.argmax(areas)) + 1
                            max_area = int(stats[max_idx, cv2.CC_STAT_AREA])

                            dominant_mask = (labels == max_idx).astype(np.uint8) * 255
                            contours, _ = cv2.findContours(dominant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            max_contour = contours[0] if contours else None

                            shape_label = analyze_shape(max_contour, max_area)
                            size_label = analyze_size(total_tumor_area, h, w)
                            location_label = analyze_location(binary, h, w)

                            semantic_data = {
                                "size": size_label,
                                "shape": shape_label,
                                "spread": spread_label,
                                "location": location_label,
                            }
                            processed_count += 1

        except Exception as e:
            if verbose:
                sid = entry.get("sample_id", f"index_{idx}")
                print(f"⚠️ Error processing {sid}: {e}")

        entry["medical_semantics"] = semantic_data
        if semantic_data is None:
            null_count += 1

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("\n✅ Finished.")
    print("📊 Stats:")
    print(f"   - Valid semantics (Object): {processed_count}")
    print(f"   - Null semantics:          {null_count}")


def main():
    parser = argparse.ArgumentParser(description="Enrich metadata with medical semantics derived from segmentation masks.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=os.getenv("DATASET_ROOT", "./data"),
        help="Dataset root directory used to resolve mask_path. Default: $DATASET_ROOT or ./data",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.getenv("INPUT_JSON", "./outputs/all_brain_tumor_metadata.json"),
        help="Input merged metadata JSON. Default: $INPUT_JSON or ./outputs/all_brain_tumor_metadata.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.getenv("OUTPUT_JSON", "./outputs/all_brain_tumor_metadata_rich.json"),
        help="Output enriched JSON. Default: $OUTPUT_JSON or ./outputs/all_brain_tumor_metadata_rich.json",
    )
    parser.add_argument(
        "--keep_origin_json",
        action="store_true",
        help="Keep origin_json field (default drops it).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample warnings on failures.",
    )
    args = parser.parse_args()

    process_medical_semantics(
        dataset_root=args.dataset_root,
        input_json=args.input,
        output_json=args.output,
        drop_origin_json=(not args.keep_origin_json),
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
