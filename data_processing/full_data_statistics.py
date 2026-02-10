import os
import json
import argparse
from typing import Any, Dict, List
import pandas as pd

# ✅ Priority list (specific > general). Put fallback last.
DEFAULT_JSON_FILES = [
    "metadata_tumor_05.json",   # 05 (Brisc)
    "metadata_10.json",         # 10 (Mask)
    "metadata_16.json",         # 16 (BraTS)
    "metadata_tumor_14.json",   # 14 (T1CE)
    "metadata_tumor_19.json",   # 19 (YOLO)
    "metadata_no_mask.json",    # fallback (no mask)
]


def safe_makedirs(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data)}")
    return data


def normalize_path(p: str) -> str:
    # Normalize separators, remove redundant "..", etc.
    return os.path.normpath(p)


def make_relative(p: str, base: str) -> str:
    # Convert to relative path when possible
    try:
        return os.path.relpath(p, base)
    except Exception:
        return p


def merge_data(
    input_dir: str,
    output_dir: str,
    json_files: List[str],
    output_json: str,
    output_stats: str,
    *,
    make_paths_relative: bool = True,
) -> None:
    print("🚀 Merging dataset metadata (no global CSV for samples)...")
    print(f"📥 input_dir:  {input_dir}")
    print(f"📤 output_dir: {output_dir}")

    safe_makedirs(output_dir)

    all_data: List[Dict[str, Any]] = []
    seen_paths = set()

    for json_file in json_files:
        json_path = os.path.join(input_dir, json_file)
        if not os.path.exists(json_path):
            print(f"⚠️ Skip (not found): {json_path}")
            continue

        data = load_json(json_path)
        print(f"📖 Loaded {json_file}: {len(data)} entries")

        count_added = 0
        count_skipped = 0

        for entry in data:
            if "image_path" not in entry:
                # Skip malformed entry
                count_skipped += 1
                continue

            path = normalize_path(str(entry["image_path"]))

            # Optional: strip absolute paths from outputs (recommended for open-sourcing)
            if make_paths_relative:
                path = make_relative(path, input_dir)
                entry["image_path"] = path

            # Deduplication
            if path in seen_paths:
                count_skipped += 1
                continue

            seen_paths.add(path)

            # Record source (clean tag)
            entry["origin"] = os.path.splitext(os.path.basename(json_file))[0]
            all_data.append(entry)
            count_added += 1

        print(f"   └─ ✅ added: {count_added} | 🚫 deduped/skipped: {count_skipped}")

    # Save merged JSON
    merged_json_path = os.path.join(output_dir, output_json)
    print(f"\n💾 Writing merged JSON: {merged_json_path}")
    with open(merged_json_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"🎉 Done! Total merged entries: {len(all_data)}")

    # Stats
    stats_path = os.path.join(output_dir, output_stats)
    print(f"\n📊 Writing stats CSV: {stats_path}")

    df = pd.DataFrame(all_data)

    stats_list: List[Dict[str, Any]] = []
    stats_list.append({"Dimension": "Overall", "Category": "Total Images", "Count": int(len(df))})

    # robust has_mask
    if "has_mask" in df.columns:
        has_mask_series = df["has_mask"].fillna(False).astype(bool)
        stats_list.append({"Dimension": "Overall", "Category": "With Mask", "Count": int(has_mask_series.sum())})
    else:
        stats_list.append({"Dimension": "Overall", "Category": "With Mask", "Count": 0})

    if "tumor_type" in df.columns:
        for name, count in df["tumor_type"].value_counts(dropna=False).items():
            stats_list.append({"Dimension": "Tumor Type", "Category": str(name), "Count": int(count)})

    if "modality" in df.columns:
        for name, count in df["modality"].value_counts(dropna=False).items():
            stats_list.append({"Dimension": "MRI Modality", "Category": str(name), "Count": int(count)})

    if "origin" in df.columns:
        for name, count in df["origin"].value_counts().items():
            stats_list.append({"Dimension": "Source", "Category": str(name), "Count": int(count)})

    df_stats = pd.DataFrame(stats_list)
    df_stats.to_csv(stats_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 44)
    print("📊 Dataset Statistics Summary")
    print("=" * 44)
    print(df_stats.to_string(index=False))
    print("=" * 44)
    print("✅ All finished.")


def main():
    parser = argparse.ArgumentParser(description="Merge multiple dataset metadata JSONs with priority & dedup.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.getenv("METADATA_DIR", "./metadata"),
        help="Directory containing metadata_*.json files. Default: $METADATA_DIR or ./metadata",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.getenv("OUTPUT_DIR", "./outputs"),
        help="Directory to save merged outputs. Default: $OUTPUT_DIR or ./outputs",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="all_brain_tumor_metadata.json",
        help="Merged JSON file name.",
    )
    parser.add_argument(
        "--output_stats",
        type=str,
        default="dataset_statistics.csv",
        help="Statistics CSV file name.",
    )
    parser.add_argument(
        "--keyword_files",
        type=str,
        default="",
        help="Optional comma-separated JSON file list to override defaults.",
    )
    parser.add_argument(
        "--keep_absolute_paths",
        action="store_true",
        help="Keep absolute image_path as-is (NOT recommended for open source).",
    )
    args = parser.parse_args()

    if args.keyword_files.strip():
        json_files = [x.strip() for x in args.keyword_files.split(",") if x.strip()]
    else:
        json_files = DEFAULT_JSON_FILES

    merge_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        json_files=json_files,
        output_json=args.output_json,
        output_stats=args.output_stats,
        make_paths_relative=(not args.keep_absolute_paths),
    )


if __name__ == "__main__":
    main()
