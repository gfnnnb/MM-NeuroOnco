import os
import json
import cv2
import re
import argparse
import hashlib
from tqdm import tqdm

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

MODALITY_PATTERNS = [
    (r"t1ce|t1\+c|gd", "t1ce"),
    (r"flair", "flair"),
    (r"t1", "t1"),
    (r"t2", "t2"),
]


def get_modality(text: str) -> str:
    text_lower = text.lower()
    for pattern, modality in MODALITY_PATTERNS:
        if re.search(pattern, text_lower):
            return modality
    return "unknown"


def clean_token(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "na"


def stable_hash(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def read_image_meta(path: str):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) > 2 else 1
    return int(h), int(w), int(c)


def should_skip_root(root: str, skip_dir_keywords: list[str]) -> bool:
    # Skip if any path component matches keyword (case-insensitive)
    parts = [p.lower() for p in root.split(os.sep) if p]
    for kw in skip_dir_keywords:
        kw = kw.lower().strip()
        if not kw:
            continue
        if kw in parts:
            return True
    return False


def process_no_mask_data(
    dataset_root: str,
    output_json: str,
    *,
    anonymize_ids: bool = False,
    skip_dir_keywords: list[str] | None = None,
    verbose: bool = False,
) -> None:
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if skip_dir_keywords is None:
        # safer default than substring "mask" everywhere
        skip_dir_keywords = ["mask", "masks", "label", "labels"]

    categories = [
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ]

    print("🚀 Extracting NO-mask images metadata...")
    print(f"📁 dataset_root: {dataset_root}")
    print(f"📌 categories: {len(categories)}")
    print(f"⛔ skip_dir_keywords: {skip_dir_keywords}")
    print(f"🕵️ anonymize_ids: {anonymize_ids}")

    metadata_list = []

    for category in tqdm(categories, desc="Scanning Categories"):
        class_path = os.path.join(dataset_root, category)

        for root, _, files in os.walk(class_path):
            # Skip directories by component match
            if should_skip_root(root, skip_dir_keywords):
                continue

            for file in files:
                low = file.lower()
                if not low.endswith(VALID_EXTS):
                    continue
                if "mask" in low:
                    continue

                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, dataset_root)

                modality = get_modality(file)

                file_name_no_ext = os.path.splitext(file)[0]
                clean_name = clean_token(file_name_no_ext)
                clean_cat = clean_token(category)

                if clean_cat.lower() in clean_name.lower():
                    raw_sample_id = clean_name
                else:
                    raw_sample_id = f"{clean_cat}_{clean_name}"

                raw_sample_id = re.sub(r"_+", "_", raw_sample_id).strip("_")

                if anonymize_ids:
                    sample_id = f"s_{stable_hash(raw_sample_id)}"
                    tumor_type = f"c_{stable_hash(category, 8)}"
                else:
                    sample_id = raw_sample_id
                    tumor_type = category

                meta = read_image_meta(abs_path)
                if meta is None:
                    if verbose:
                        print(f"[WARN] Failed to read image: {abs_path}")
                    continue
                height, width, channel = meta

                entry = {
                    "sample_id": sample_id,
                    "image_path": rel_path,
                    "tumor_present": (category.lower() != "healthy"),
                    "tumor_type": tumor_type,
                    "modality": modality,
                    "has_mask": False,
                    "mask_path": "",
                    "image_meta": {"height": height, "width": width, "channel": channel},
                }
                metadata_list.append(entry)

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    print(f"💾 Saving {len(metadata_list)} entries to {output_json} ...")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    print("✅ Done.")


def main():
    parser = argparse.ArgumentParser(description="Extract metadata for images WITHOUT masks (open-source safe).")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=os.getenv("DATASET_ROOT", "./data/no_mask_dataset"),
        help="Dataset root directory. Default: $DATASET_ROOT or ./data/no_mask_dataset",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=os.getenv("OUTPUT_JSON", "./outputs/metadata_no_mask.json"),
        help="Output metadata JSON path. Default: $OUTPUT_JSON or ./outputs/metadata_no_mask.json",
    )
    parser.add_argument(
        "--anonymize_ids",
        action="store_true",
        help="Anonymize sample_id and tumor_type derived from folder/file names (recommended for medical data).",
    )
    parser.add_argument(
        "--skip_dir_keywords",
        type=str,
        default=os.getenv("SKIP_DIR_KEYWORDS", "mask,masks,label,labels"),
        help="Comma-separated directory names to skip. Default: mask,masks,label,labels",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print warnings for unreadable images.",
    )
    args = parser.parse_args()

    skip_keywords = [x.strip() for x in args.skip_dir_keywords.split(",") if x.strip()]

    process_no_mask_data(
        dataset_root=args.dataset_root,
        output_json=args.output_json,
        anonymize_ids=args.anonymize_ids,
        skip_dir_keywords=skip_keywords,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
