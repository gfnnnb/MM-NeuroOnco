import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm

# Supported image extensions
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def safe_makedirs(path: str) -> None:
    """Create directory if path is non-empty."""
    if path:
        os.makedirs(path, exist_ok=True)


def convert_txt_to_mask(
    img_path: str,
    txt_path: str,
    save_path: str,
    *,
    verbose: bool = False,
    strict: bool = False,
) -> bool:
    """Read image size + YOLO-seg txt polygon coords, generate a binary mask image."""
    img = cv2.imread(img_path)
    if img is None:
        if verbose:
            print(f"[WARN] Failed to read image: {img_path}")
        return False

    h, w = img.shape[:2]

    if not os.path.exists(txt_path):
        if verbose:
            print(f"[WARN] Label txt not found: {txt_path}")
        return False

    mask = np.zeros((h, w), dtype=np.uint8)

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    has_tumor = False
    for line_idx, line in enumerate(lines):
        parts = line.strip().split()
        # YOLOv8 seg format: class x1 y1 x2 y2 ... (>= class + 3 points => 7 tokens)
        if len(parts) < 7:
            if verbose:
                print(f"[WARN] Invalid seg line (too short) at {txt_path}:{line_idx + 1}")
            if strict:
                return False
            continue

        try:
            normalized_coords = [float(x) for x in parts[1:]]
            if len(normalized_coords) % 2 != 0:
                raise ValueError("Odd number of coordinates")

            points = np.array(normalized_coords, dtype=np.float32).reshape(-1, 2)
            points[:, 0] *= w
            points[:, 1] *= h

            # Optional: clip to image bounds to avoid weird coords
            points[:, 0] = np.clip(points[:, 0], 0, w - 1)
            points[:, 1] = np.clip(points[:, 1], 0, h - 1)

            points = points.astype(np.int32)

            cv2.fillPoly(mask, [points], 255)
            has_tumor = True

        except Exception as e:
            if verbose:
                print(f"[WARN] Failed parsing line at {txt_path}:{line_idx + 1} ({e})")
            if strict:
                return False
            continue

    if has_tumor:
        safe_makedirs(os.path.dirname(save_path))
        ok = cv2.imwrite(save_path, mask)
        if not ok and verbose:
            print(f"[WARN] Failed to write mask: {save_path}")
        return ok

    return False


def infer_labels_dir(images_dir: str, labels_dirname: str = "labels") -> str:
    """
    Infer labels directory from an images directory.
    Supports structures like:
      .../train/images  -> .../train/labels
      .../images        -> .../labels
    """
    parent = os.path.dirname(images_dir)
    return os.path.join(parent, labels_dirname)


def batch_convert(
    dataset_root: str,
    keyword: str,
    *,
    images_dirname: str = "images",
    labels_dirname: str = "labels",
    output_dir: str | None = None,
    verbose: bool = False,
    strict: bool = False,
) -> None:
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    subfolders = [
        d for d in os.listdir(dataset_root)
        if keyword in d and os.path.isdir(os.path.join(dataset_root, d))
    ]

    if not subfolders:
        print(f"❌ No subfolders containing keyword '{keyword}' under: {dataset_root}")
        return

    print(f"🚀 Found target folders: {subfolders}")

    stats = {
        "found_images": 0,
        "missing_labels_dir": 0,
        "missing_txt": 0,
        "converted": 0,
        "skipped_no_tumor": 0,
    }

    for sub in subfolders:
        base_path = os.path.join(dataset_root, sub)

        for root, _, files in os.walk(base_path):
            if os.path.basename(root) != images_dirname:
                continue

            images_dir = root
            labels_dir = infer_labels_dir(images_dir, labels_dirname=labels_dirname)

            if not os.path.exists(labels_dir):
                stats["missing_labels_dir"] += 1
                if verbose:
                    print(f"[WARN] labels folder not found: {labels_dir} (images_dir={images_dir})")
                continue

            image_files = [
                f for f in files
                if f.lower().endswith(VALID_EXTS) and "mask" not in f.lower()
            ]
            stats["found_images"] += len(image_files)

            print(f"📂 Processing: {images_dir} ({len(image_files)} images)")

            # Decide where to save masks
            if output_dir is None:
                out_images_dir = images_dir
            else:
                # mirror relative path under dataset_root
                rel = os.path.relpath(images_dir, dataset_root)
                out_images_dir = os.path.join(output_dir, rel)
                safe_makedirs(out_images_dir)

            for img_file in tqdm(image_files, desc=os.path.basename(os.path.dirname(images_dir))):
                name_no_ext = os.path.splitext(img_file)[0]
                img_path = os.path.join(images_dir, img_file)
                txt_path = os.path.join(labels_dir, f"{name_no_ext}.txt")
                save_path = os.path.join(out_images_dir, f"{name_no_ext}_mask.png")

                if not os.path.exists(txt_path):
                    stats["missing_txt"] += 1
                    if verbose:
                        print(f"[WARN] Missing txt: {txt_path}")
                    if strict:
                        raise FileNotFoundError(f"Missing txt: {txt_path}")
                    continue

                ok = convert_txt_to_mask(
                    img_path, txt_path, save_path,
                    verbose=verbose, strict=strict
                )
                if ok:
                    stats["converted"] += 1
                else:
                    stats["skipped_no_tumor"] += 1

    print("\n✅ Done!")
    print("📊 Summary:")
    for k, v in stats.items():
        print(f"  - {k}: {v}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO-seg .txt polygons to binary mask PNGs (open-source safe)."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=os.getenv("DATASET_ROOT", "./data/images"),
        help="Root directory that contains raw datasets. Defaults to $DATASET_ROOT or ./data/images",
    )
    parser.add_argument(
        "--keyword",
        type=str,
        default="19",
        help="Only process subfolders whose name contains this keyword (default: '19').",
    )
    parser.add_argument(
        "--images_dirname",
        type=str,
        default="images",
        help="Directory name that contains images (default: images).",
    )
    parser.add_argument(
        "--labels_dirname",
        type=str,
        default="labels",
        help="Directory name that contains YOLO labels (default: labels).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.getenv("OUTPUT_DIR", ""),
        help="Optional output root to write masks. If empty, writes next to images. "
             "Defaults to $OUTPUT_DIR or empty.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print warnings for missing/invalid files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast on invalid labels or missing txt.",
    )
    args = parser.parse_args()

    out_dir = args.output_dir.strip() or None

    batch_convert(
        args.dataset_root,
        args.keyword,
        images_dirname=args.images_dirname,
        labels_dirname=args.labels_dirname,
        output_dir=out_dir,
        verbose=args.verbose,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
