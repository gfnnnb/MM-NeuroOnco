import os
import json
import cv2
import re
import argparse
import hashlib
from tqdm import tqdm

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Modality patterns (longer / more specific first)
MODALITY_PATTERNS = [
    (r"t1ce|t1\+c|gd", "t1ce"),
    (r"flair", "flair"),
    (r"t1", "t1"),
    (r"t2", "t2"),
]


def get_modality(filename: str) -> str:
    name_lower = filename.lower()
    for pattern, modality in MODALITY_PATTERNS:
        if re.search(pattern, name_lower):
            return modality
    return "unknown"


def clean_token(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "na"


def stable_hash(text: str, length: int = 10) -> str:
    # Stable anonymized token (deterministic, no secret)
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return h[:length]


def read_image_meta(path: str) -> dict:
    img = cv2.imread(path)
    if img is None:
        return {"height": 0, "width": 0, "channel": 0}
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) > 2 else 1
    return {"height": int(h), "width": int(w), "channel": int(c)}


def find_mask_file(file_set: set[str], name_no_ext: str) -> str | None:
    """
    More robust mask matching:
    - allow mask extension differ from image extension
    - match <name>_mask.<ext> where ext in VALID_EXTS
    """
    for ext in VALID_EXTS:
        candidate = f"{name_no_ext}_mask{ext}"
        if candidate in file_set:
            return candidate
    return None


def process_dataset(
    dataset_root: str,
    output_json: str,
    *,
    target_keyword: str = "10",
    anonymize_labels: bool = False,
    verbose: bool = False,
) -> None:
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    metadata_list = []

    categories = [
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ]
    print(f"🚀 Scanning dataset (keyword={target_keyword})")
    print(f"📁 dataset_root: {dataset_root}")
    print(f"📌 categories: {len(categories)}")

    for category in tqdm(categories, desc="Scanning Categories"):
        class_path = os.path.join(dataset_root, category)

        subfolders = [
            d for d in os.listdir(class_path)
            if os.path.isdir(os.path.join(class_path, d))
        ]
        target_subfolders = [d for d in subfolders if target_keyword in d]

        for subfolder in target_subfolders:
            source_path = os.path.join(class_path, subfolder)
            files = os.listdir(source_path)
            file_set = set(files)

            for file in files:
                if not file.lower().endswith(VALID_EXTS):
                    continue

                name_no_ext, _ = os.path.splitext(file)

                # skip mask images themselves
                if name_no_ext.lower().endswith("_mask"):
                    continue

                # mask detection (robust to extension)
                mask_file = find_mask_file(file_set, name_no_ext)
                has_mask = mask_file is not None

                mask_rel_path = ""
                if has_mask:
                    mask_abs_path = os.path.join(source_path, mask_file)
                    mask_rel_path = os.path.relpath(mask_abs_path, dataset_root)

                modality = get_modality(file)

                # ---- sample_id ----
                clean_name = clean_token(name_no_ext)
                clean_cat = clean_token(category)
                clean_sub = clean_token(subfolder)

                if clean_cat.lower() in clean_name.lower():
                    sample_id = f"{clean_sub}_{clean_name}"
                else:
                    sample_id = f"{clean_cat}_{clean_sub}_{clean_name}"

                # optional anonymization (for open-source safety)
                if anonymize_labels:
                    # keep structure but hide original tokens
                    sample_id = f"s_{stable_hash(sample_id, 12)}"
                    tumor_type = f"c_{stable_hash(category, 8)}"
                else:
                    tumor_type = category

                # paths (store relative to dataset_root)
                abs_path = os.path.join(source_path, file)
                rel_path = os.path.relpath(abs_path, dataset_root)

                image_meta = read_image_meta(abs_path)
                if image_meta["height"] == 0 and verbose:
                    print(f"[WARN] Failed to read image: {abs_path}")

                entry = {
                    "sample_id": sample_id,
                    "image_path": rel_path,
                    "tumor_present": (category.lower() != "healthy"),
                    "tumor_type": tumor_type,
                    "modality": modality,
                    "has_mask": has_mask,
                    "mask_path": mask_rel_path,
                    "image_meta": image_meta,
                }
                metadata_list.append(entry)

    # write
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved metadata: {output_json}")
    print(f"📊 Total entries: {len(metadata_list)}")

    # modality summary
    mod_counts = {}
    for x in metadata_list:
        m = x["modality"]
        mod_counts[m] = mod_counts.get(m, 0) + 1
    print("\n[Modality Stats]")
    print(json.dumps(mod_counts, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Generate metadata JSON for dataset (open-source safe).")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=os.getenv("DATASET_ROOT", "./data/dataset_10"),
        help="Dataset root directory. Default: $DATASET_ROOT or ./data/dataset_10",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=os.getenv("OUTPUT_JSON", "./outputs/metadata_10.json"),
        help="Output metadata json path. Default: $OUTPUT_JSON or ./outputs/metadata_10.json",
    )
    parser.add_argument(
        "--keyword",
        type=str,
        default="10",
        help="Only scan subfolders whose name contains this keyword (default: 10).",
    )
    parser.add_argument(
        "--anonymize_labels",
        action="store_true",
        help="Anonymize tumor_type and sample_id tokens (useful when folder names are private).",
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
        target_keyword=args.keyword,
        anonymize_labels=args.anonymize_labels,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
