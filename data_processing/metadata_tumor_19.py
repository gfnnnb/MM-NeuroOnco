import os
import json
import cv2
import re
import argparse
import hashlib
from tqdm import tqdm

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# YOLO class map
CLASS_MAP = {
    0: "glioma",
    1: "meningioma",
    2: "pituitary",
}


def clean_token(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "na"


def stable_hash(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def path_has_component(path: str, name: str) -> bool:
    parts = [p.lower() for p in path.split(os.sep) if p]
    return name.lower() in parts


def infer_labels_dir(images_dir: str) -> str:
    """
    Infer labels directory from an images directory.
    Supports:
      .../train/images -> .../train/labels
      .../images       -> .../labels
    """
    parent = os.path.dirname(images_dir)
    return os.path.join(parent, "labels")


def get_tumor_type(txt_path: str, *, verbose: bool = False) -> str:
    """Read YOLO txt first token class id -> tumor type."""
    if not os.path.exists(txt_path):
        return "tumor"

    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            line = f.readline()
        if not line:
            return "tumor"
        parts = line.strip().split()
        if not parts:
            return "tumor"
        class_id = int(parts[0])
        return CLASS_MAP.get(class_id, "tumor")
    except Exception as e:
        if verbose:
            print(f"[WARN] Failed to parse txt: {txt_path} ({e})")
        return "tumor"


def read_image_meta(path: str):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) > 2 else 1
    return int(h), int(w), int(c)


def guess_modality_from_name(filename: str) -> str:
    low = filename.lower()
    if "t1ce" in low:
        return "t1ce"
    if "flair" in low:
        return "flair"
    if "t1" in low:
        return "t1"
    if "t2" in low:
        return "t2"
    return "unknown"


def process_dataset_19(
    dataset_root: str,
    output_json: str,
    *,
    tumor_folder: str = "tumor",
    keyword: str = "19",
    anonymize_ids: bool = False,
    verbose: bool = False,
) -> None:
    tumor_dir = os.path.join(dataset_root, tumor_folder)
    if not os.path.exists(tumor_dir):
        raise FileNotFoundError(f"tumor folder not found: {tumor_dir}")

    subfolders = [
        d for d in os.listdir(tumor_dir)
        if keyword in d and os.path.isdir(os.path.join(tumor_dir, d))
    ]
    if not subfolders:
        raise FileNotFoundError(f"No subfolder containing '{keyword}' under: {tumor_dir}")

    print(f"🚀 Extracting dataset_19 metadata (keyword={keyword})")
    print(f"📁 dataset_root: {dataset_root}")
    print(f"📌 target folders: {subfolders}")
    print(f"🕵️ anonymize_ids: {anonymize_ids}")

    metadata_list = []
    counts = {}
    total_scanned = 0
    total_with_mask = 0

    for subfolder in subfolders:
        base_path = os.path.join(tumor_dir, subfolder)

        for root, _, files in os.walk(base_path):
            # Skip labels dirs (component-wise, cross-platform)
            if path_has_component(root, "labels"):
                continue

            # Only handle images dirs if you want strict structure:
            # if os.path.basename(root).lower() != "images": continue
            # Here we keep your original behavior (any dir except labels).

            image_files = [f for f in files if f.lower().endswith(VALID_EXTS)]
            if not image_files:
                continue

            mask_files = {f for f in image_files if "_mask" in f.lower()}
            original_images = [f for f in image_files if "_mask" not in f.lower()]
            if not original_images:
                continue

            # Infer labels dir from current folder
            labels_dir = infer_labels_dir(root)
            if not os.path.exists(labels_dir):
                # fallback: sibling labels
                labels_dir = os.path.join(os.path.dirname(root), "labels")

            for img_file in tqdm(original_images, desc=f"Scanning {os.path.basename(root)}"):
                total_scanned += 1

                name_no_ext = os.path.splitext(img_file)[0]
                abs_img_path = os.path.join(root, img_file)

                expected_mask = f"{name_no_ext}_mask.png"
                has_mask = expected_mask in mask_files
                mask_rel_path = ""

                if has_mask:
                    total_with_mask += 1
                    mask_abs_path = os.path.join(root, expected_mask)
                    mask_rel_path = os.path.relpath(mask_abs_path, dataset_root)

                # tumor type from txt
                txt_name = f"{name_no_ext}.txt"
                abs_txt_path = os.path.join(labels_dir, txt_name)
                final_tumor_type = get_tumor_type(abs_txt_path, verbose=verbose)

                # sample_id
                rel_dir = os.path.relpath(root, tumor_dir)
                clean_dirs = clean_token(rel_dir)
                clean_name = clean_token(name_no_ext)
                raw_id = f"tumor_{clean_dirs}_{clean_name}"
                raw_id = re.sub(r"_+", "_", raw_id).strip("_")

                sample_id = f"s_{stable_hash(raw_id)}" if anonymize_ids else raw_id

                modality = guess_modality_from_name(img_file)

                meta = read_image_meta(abs_img_path)
                if meta is None:
                    if verbose:
                        print(f"[WARN] Failed to read image: {abs_img_path}")
                    continue
                h, w, c = meta

                entry = {
                    "sample_id": sample_id,
                    "image_path": os.path.relpath(abs_img_path, dataset_root),
                    "tumor_present": True,
                    "tumor_type": final_tumor_type,
                    "modality": modality,
                    "has_mask": has_mask,
                    "mask_path": mask_rel_path,
                    "image_meta": {"height": h, "width": w, "channel": c},
                }
                metadata_list.append(entry)

                counts[final_tumor_type] = counts.get(final_tumor_type, 0) + 1

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved: {output_json}")
    print(f"📊 Total metadata entries: {len(metadata_list)}")
    if total_scanned:
        print(f"🩻 Mask coverage (scanned): {total_with_mask}/{total_scanned} = {total_with_mask/total_scanned:.2%}")

    print("\n[Class distribution]")
    print(json.dumps(counts, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Extract YOLO-based dataset_19 metadata (open-source safe).")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=os.getenv("DATASET_ROOT", "./data/dataset_19"),
        help="Dataset root directory. Default: $DATASET_ROOT or ./data/dataset_19",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=os.getenv("OUTPUT_JSON", "./outputs/metadata_tumor_19.json"),
        help="Output metadata JSON path. Default: $OUTPUT_JSON or ./outputs/metadata_tumor_19.json",
    )
    parser.add_argument(
        "--tumor_folder",
        type=str,
        default=os.getenv("TUMOR_FOLDER", "tumor"),
        help="Top-level tumor folder name. Default: tumor",
    )
    parser.add_argument(
        "--keyword",
        type=str,
        default="19",
        help="Only process subfolders containing this keyword (default: 19).",
    )
    parser.add_argument(
        "--anonymize_ids",
        action="store_true",
        help="Anonymize sample_id derived from folder/file names (recommended for medical data).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print warnings for unreadable images / label parse failures.",
    )
    args = parser.parse_args()

    process_dataset_19(
        dataset_root=args.dataset_root,
        output_json=args.output_json,
        tumor_folder=args.tumor_folder,
        keyword=args.keyword,
        anonymize_ids=args.anonymize_ids,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
