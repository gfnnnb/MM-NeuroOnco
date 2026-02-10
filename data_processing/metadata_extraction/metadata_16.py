import os
import json
import cv2
import re
import argparse
import hashlib
from tqdm import tqdm

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# BraTS modality suffix map
MODALITY_MAP = {
    "t1": "t1",
    "t2": "t2",
    "t1ce": "t1ce",
    "flair": "flair",
}


def safe_makedirs(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def clean_token(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "na"


def stable_hash(text: str, length: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def read_image_meta(path: str) -> dict:
    img = cv2.imread(path)
    if img is None:
        return {"height": 0, "width": 0, "channel": 0}
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) > 2 else 1
    return {"height": int(h), "width": int(w), "channel": int(c)}


def infer_modality_from_name(name_no_ext: str) -> str:
    # Prefer suffix splitting
    parts = name_no_ext.split("_")
    suffix = parts[-1].lower() if parts else ""
    modality = MODALITY_MAP.get(suffix, "unknown")

    # Fallback (contains)
    if modality == "unknown":
        low = name_no_ext.lower()
        if "flair" in low:
            return "flair"
        if "t1ce" in low or "t1ce" in low:
            return "t1ce"
        if "t1" in low:
            return "t1"
        if "t2" in low:
            return "t2"
    return modality


def process_dataset(
    dataset_root: str,
    output_json: str,
    *,
    category: str = "glioma",
    keyword: str = "16",
    anonymize_labels: bool = False,
    verbose: bool = False,
) -> None:
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    class_path = os.path.join(dataset_root, category)
    if not os.path.exists(class_path):
        raise FileNotFoundError(f"Category folder not found: {class_path}")

    subfolders = [
        d for d in os.listdir(class_path)
        if os.path.isdir(os.path.join(class_path, d))
    ]
    target_subfolders = [d for d in subfolders if keyword in d]

    print(f"🚀 Processing dataset (keyword={keyword})")
    print(f"📁 dataset_root: {dataset_root}")
    print(f"🏷️ category: {category}")
    print(f"📌 target folders: {target_subfolders}")

    metadata_list = []

    for subfolder in target_subfolders:
        dataset_path = os.path.join(class_path, subfolder)

        case_folders = [
            d for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ]

        for case_folder in tqdm(case_folders, desc=f"Processing {subfolder}"):
            case_path = os.path.join(dataset_path, case_folder)
            files = os.listdir(case_path)
            file_set = set(files)

            # Find seg file (mask)
            seg_file = next((f for f in files if f.endswith("_seg.png")), None)

            has_mask = seg_file is not None
            mask_rel_path = ""
            if has_mask:
                mask_abs_path = os.path.join(case_path, seg_file)
                mask_rel_path = os.path.relpath(mask_abs_path, dataset_root)

            for file in files:
                if not file.lower().endswith(VALID_EXTS):
                    continue
                if seg_file and file == seg_file:
                    continue

                name_no_ext = os.path.splitext(file)[0]
                modality = infer_modality_from_name(name_no_ext)

                abs_path = os.path.join(case_path, file)
                rel_path = os.path.relpath(abs_path, dataset_root)

                # sample_id generation
                clean_name = clean_token(name_no_ext)
                clean_cat = clean_token(category)
                clean_sub = clean_token(subfolder)

                if clean_cat.lower() in clean_name.lower():
                    raw_id = f"{clean_sub}_{clean_name}"
                else:
                    raw_id = f"{clean_cat}_{clean_sub}_{clean_name}"

                raw_id = re.sub(r"_+", "_", raw_id).strip("_")

                if anonymize_labels:
                    sample_id = f"s_{stable_hash(raw_id, 12)}"
                    tumor_type = f"c_{stable_hash(category, 8)}"
                else:
                    sample_id = raw_id
                    tumor_type = category

                image_meta = read_image_meta(abs_path)
                if verbose and image_meta["height"] == 0:
                    print(f"[WARN] Failed to read image: {abs_path}")

                entry = {
                    "sample_id": sample_id,
                    "image_path": rel_path,
                    "tumor_present": True,
                    "tumor_type": tumor_type,
                    "modality": modality,
                    "has_mask": has_mask,
                    "mask_path": mask_rel_path,
                    "image_meta": image_meta,
                }
                metadata_list.append(entry)

    safe_makedirs(os.path.dirname(output_json) or ".")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved: {output_json}")
    print(f"📊 Total images: {len(metadata_list)}")


def main():
    parser = argparse.ArgumentParser(description="Extract BraTS-like dataset metadata (open-source safe).")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=os.getenv("DATASET_ROOT", "./data/brats_like"),
        help="Dataset root directory. Default: $DATASET_ROOT or ./data/brats_like",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=os.getenv("OUTPUT_JSON", "./outputs/metadata_16.json"),
        help="Output metadata JSON path. Default: $OUTPUT_JSON or ./outputs/metadata_16.json",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=os.getenv("CATEGORY", "glioma"),
        help="Target category folder name. Default: $CATEGORY or glioma",
    )
    parser.add_argument(
        "--keyword",
        type=str,
        default="16",
        help="Only process subfolders containing this keyword (default: 16).",
    )
    parser.add_argument(
        "--anonymize_labels",
        action="store_true",
        help="Anonymize sample_id/tumor_type tokens (recommended if folder names are private).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print warnings for unreadable images.",
    )
    args = parser.parse_args()

    process_dataset(
        dataset_root=args.dataset_root,
        output_json=args.output_json,
        category=args.category,
        keyword=args.keyword,
        anonymize_labels=args.anonymize_labels,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
