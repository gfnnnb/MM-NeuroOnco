import os
import json
import cv2
import re
import argparse
import hashlib
from tqdm import tqdm

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


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


def extract_id_from_mask_filename(fname: str) -> str | None:
    """
    Extract id from something like: "mask  (1644).png"
    Prefer digits inside parentheses; fallback to any text inside parentheses.
    """
    m = re.search(r"\((\d+)\)", fname)
    if m:
        return m.group(1)
    m = re.search(r"\((.+?)\)", fname)
    if m:
        return m.group(1).strip()
    return None


def process_dataset_14(
    dataset_root: str,
    output_json: str,
    *,
    tumor_folder: str = "tumor",
    keyword: str = "14",
    modality: str = "t1ce",
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

    print(f"🚀 Found target folders: {subfolders}")

    metadata_list = []
    total_matched = 0
    total_images = 0

    for subfolder in subfolders:
        target_dir = os.path.join(tumor_dir, subfolder)
        files = os.listdir(target_dir)

        # Build mask map: key=id in parentheses, value=filename
        mask_map = {}
        for f in files:
            low = f.lower()
            if "mask" in low and low.endswith(VALID_EXTS):
                file_id = extract_id_from_mask_filename(f)
                if file_id:
                    mask_map[file_id] = f

        print(f"✅ [{subfolder}] built {len(mask_map)} mask entries")

        image_files = [
            f for f in files
            if f.lower().endswith(VALID_EXTS) and "mask" not in f.lower()
        ]

        for img_file in tqdm(image_files, desc=f"Processing {subfolder}"):
            name_no_ext, _ = os.path.splitext(img_file)  # often "1644"
            total_images += 1

            mask_name = mask_map.get(name_no_ext)
            has_mask = mask_name is not None
            mask_rel_path = ""

            if has_mask:
                total_matched += 1
                mask_abs_path = os.path.join(target_dir, mask_name)
                mask_rel_path = os.path.relpath(mask_abs_path, dataset_root)

            abs_path = os.path.join(target_dir, img_file)
            rel_path = os.path.relpath(abs_path, dataset_root)

            clean_sub = clean_token(subfolder)
            clean_name = clean_token(name_no_ext)
            raw_id = f"tumor_{clean_sub}_{clean_name}"

            sample_id = f"s_{stable_hash(raw_id)}" if anonymize_ids else raw_id

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
                "tumor_type": "tumor",
                "modality": modality,
                "has_mask": has_mask,
                "mask_path": mask_rel_path,
                "image_meta": {"height": h, "width": w, "channel": c},
            }
            metadata_list.append(entry)

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    # Note: metadata_list may exclude unreadable images; total_images counts scanned files
    print(f"\n✅ Saved: {output_json}")
    print(f"📊 Metadata entries: {len(metadata_list)}")
    if total_images > 0:
        print(f"🔍 Mask match rate (scanned images): {total_matched}/{total_images} = {total_matched/total_images:.2%}")
    else:
        print("🔍 Mask match rate: N/A (no images found)")


def main():
    parser = argparse.ArgumentParser(description="Extract dataset_14 metadata with mask mapping (open-source safe).")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=os.getenv("DATASET_ROOT", "./data/dataset_14"),
        help="Dataset root directory. Default: $DATASET_ROOT or ./data/dataset_14",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=os.getenv("OUTPUT_JSON", "./outputs/metadata_tumor_14.json"),
        help="Output metadata JSON path. Default: $OUTPUT_JSON or ./outputs/metadata_tumor_14.json",
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
        default="14",
        help="Only process subfolders containing this keyword (default: 14).",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default=os.getenv("MODALITY", "t1ce"),
        help="Modality to write into metadata (default: t1ce).",
    )
    parser.add_argument(
        "--anonymize_ids",
        action="store_true",
        help="Anonymize sample_id derived from folder/file names (recommended for medical data).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print warnings for unreadable images.",
    )
    args = parser.parse_args()

    process_dataset_14(
        dataset_root=args.dataset_root,
        output_json=args.output_json,
        tumor_folder=args.tumor_folder,
        keyword=args.keyword,
        modality=args.modality,
        anonymize_ids=args.anonymize_ids,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
