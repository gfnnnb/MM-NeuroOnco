import os
import json
import cv2
import re
import argparse
import hashlib
from tqdm import tqdm

VALID_IMG_EXTS = (".jpg", ".jpeg")  # dataset_05 only images are jpg/jpeg
MASK_EXT = ".png"

# Tumor type abbreviation map (dataset-specific)
TYPE_MAP = {
    "gl": "glioma",
    "me": "meningioma",
    "pi": "pituitary",
    "no": "healthy",
    "tu": "tumor",
}

MODALITY_PATTERNS = [
    (r"t1ce|t1\+c", "t1ce"),
    (r"flair", "flair"),
    (r"t1", "t1"),
    (r"t2", "t2"),
]


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


def parse_filename_info(filename_no_ext: str):
    """Parse filename to get tumor_type and modality."""
    parts = filename_no_ext.split("_")
    tumor_type = "tumor"
    modality = "unknown"

    # modality: scan from tail (more specific usually at end)
    for part in reversed(parts):
        part_lower = part.lower()
        for pattern, mod in MODALITY_PATTERNS:
            if re.search(pattern, part_lower):
                modality = mod
                break
        if modality != "unknown":
            break

    # type: scan from head
    for part in parts:
        key = part.lower()
        if key in TYPE_MAP:
            tumor_type = TYPE_MAP[key]
            break

    return tumor_type, modality


def process_dataset_05(
    dataset_root: str,
    output_json: str,
    *,
    tumor_folder: str = "tumor",
    keyword: str = "05",
    anonymize_ids: bool = False,
    anonymize_labels: bool = False,
    verbose: bool = False,
) -> None:
    tumor_dir = os.path.join(dataset_root, tumor_folder)
    if not os.path.exists(tumor_dir):
        raise FileNotFoundError(f"tumor folder not found: {tumor_dir}")

    subfolders = [
        d for d in os.listdir(tumor_dir)
        if os.path.isdir(os.path.join(tumor_dir, d))
    ]
    target_subfolders = [d for d in subfolders if keyword in d]
    if not target_subfolders:
        raise FileNotFoundError(f"No subfolder containing '{keyword}' under: {tumor_dir}")

    print(f"🚀 Found target folders: {target_subfolders}")
    metadata_list = []

    for subfolder in target_subfolders:
        target_dir = os.path.join(tumor_dir, subfolder)
        files = os.listdir(target_dir)
        file_set = set(files)

        img_files = [f for f in files if f.lower().endswith(VALID_IMG_EXTS)]

        for img_file in tqdm(img_files, desc=f"Processing {subfolder}"):
            name_no_ext, _ = os.path.splitext(img_file)

            # mask: same stem + .png
            mask_name = f"{name_no_ext}{MASK_EXT}"
            has_mask = mask_name in file_set

            mask_rel_path = ""
            if has_mask:
                mask_abs_path = os.path.join(target_dir, mask_name)
                mask_rel_path = os.path.relpath(mask_abs_path, dataset_root)

            extracted_type, extracted_modality = parse_filename_info(name_no_ext)

            abs_path = os.path.join(target_dir, img_file)
            rel_path = os.path.relpath(abs_path, dataset_root)

            # sample_id (potentially sensitive if filenames contain IDs)
            clean_name = clean_token(name_no_ext)
            clean_type = clean_token(extracted_type)
            clean_sub = clean_token(subfolder)

            if clean_type.lower() in clean_name.lower():
                raw_id = f"{clean_sub}_{clean_name}"
            else:
                raw_id = f"{clean_type}_{clean_sub}_{clean_name}"
            raw_id = re.sub(r"_+", "_", raw_id).strip("_")

            if anonymize_ids:
                sample_id = f"s_{stable_hash(raw_id)}"
            else:
                sample_id = raw_id

            if anonymize_labels:
                tumor_type = f"c_{stable_hash(extracted_type, 8)}"
            else:
                tumor_type = extracted_type

            meta = read_image_meta(abs_path)
            if meta is None:
                if verbose:
                    print(f"[WARN] Failed to read image: {abs_path}")
                continue
            h, w, c = meta

            entry = {
                "sample_id": sample_id,
                "image_path": rel_path,
                "tumor_present": True,
                "tumor_type": tumor_type,
                "modality": extracted_modality,
                "has_mask": has_mask,
                "mask_path": mask_rel_path,
                "image_meta": {"height": h, "width": w, "channel": c},
            }
            metadata_list.append(entry)

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved: {output_json}")
    print(f"📊 Total entries: {len(metadata_list)}")


def main():
    parser = argparse.ArgumentParser(description="Extract dataset_05 metadata (open-source safe).")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=os.getenv("DATASET_ROOT", "./data/dataset_05"),
        help="Dataset root directory. Default: $DATASET_ROOT or ./data/dataset_05",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=os.getenv("OUTPUT_JSON", "./outputs/metadata_tumor_05.json"),
        help="Output metadata JSON path. Default: $OUTPUT_JSON or ./outputs/metadata_tumor_05.json",
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
        default="05",
        help="Only process subfolders containing this keyword (default: 05).",
    )
    parser.add_argument(
        "--anonymize_ids",
        action="store_true",
        help="Anonymize sample_id derived from folder/file names (recommended for medical data).",
    )
    parser.add_argument(
        "--anonymize_labels",
        action="store_true",
        help="Anonymize tumor_type labels (only if you consider label names sensitive).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print warnings for unreadable images.",
    )
    args = parser.parse_args()

    process_dataset_05(
        dataset_root=args.dataset_root,
        output_json=args.output_json,
        tumor_folder=args.tumor_folder,
        keyword=args.keyword,
        anonymize_ids=args.anonymize_ids,
        anonymize_labels=args.anonymize_labels,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
