import json
import os
import argparse
from tqdm import tqdm

# ==================== Constants ====================
CORE_FIELDS = ["shape", "margins", "texture", "enhancement", "edema", "signal_intensity"]
NULL_SET = {"null", "none", "unknown", "", "nan"}


# ==================== Utils ====================
def normalize_str(s):
    if s is None:
        return "null"
    return str(s).lower().strip().replace(" ", "").replace("-", "").replace("_", "")


def is_null(v):
    return normalize_str(v) in NULL_SET


def normalize_modality(mod_str):
    m = normalize_str(mod_str)
    if m in NULL_SET:
        return "null"
    if any(x in m for x in ["t1contrast", "t1+c", "t1ce"]):
        return "t1_contrast"
    if "t1" in m:
        return "t1"
    if "t2" in m:
        return "t2"
    if "flair" in m:
        return "flair"
    return "null"


def resolve_location_relaxed(l1, l2):
    """
    Relaxed location fusion:
      1) Prefer A if present
      2) Else use B if present
      3) Else null
    """
    n1, n2 = normalize_str(l1), normalize_str(l2)
    if n1 not in NULL_SET:
        return l1
    if n2 not in NULL_SET:
        return l2
    return "null"


def resolve_modality_with_downgrade(m1, m2, fused_signs):
    """
    Modality conflict resolution:
      - T1 vs T1_Contrast => tentatively T1_Contrast, but downgrade to T1 if enhancement is null
      - T2 vs FLAIR => FLAIR
      - T1 vs T2 => conflict (drop)
    """
    if m1 == m2:
        final_mod = m1
    elif m1 in ["null", "unknown"]:
        final_mod = m2
    elif m2 in ["null", "unknown"]:
        final_mod = m1
    elif {m1, m2} == {"t2", "flair"}:
        final_mod = "flair"
    elif {m1, m2} == {"t1", "t1_contrast"}:
        final_mod = "t1_contrast"
    else:
        return "conflict"

    # physical audit downgrade
    if final_mod == "t1_contrast":
        if is_null(fused_signs.get("enhancement")):
            return "t1"
    return final_mod


def merge_signs_with_downgrade(signs_a, signs_b):
    """
    High-precision merge:
      - If either side is null => null (drop field)
      - If equal => keep
      - signal_intensity:
          hyper + iso => iso
          hypo  + iso => iso
          hyper + hypo => null
      - enhancement/edema: conflict => Present
      - other conflicts => null
    """
    merged = {}
    degree_fields = {"enhancement", "edema"}

    for key in CORE_FIELDS:
        val_a, val_b = signs_a.get(key), signs_b.get(key)
        norm_a, norm_b = normalize_str(val_a), normalize_str(val_b)

        if norm_a in NULL_SET or norm_b in NULL_SET:
            merged[key] = "null"
            continue

        if norm_a == norm_b:
            merged[key] = val_a
            continue

        if key == "signal_intensity":
            si_set = {norm_a, norm_b}
            if si_set == {"hyperintense", "isointense"}:
                merged[key] = "Isointense"
            elif si_set == {"hypointense", "isointense"}:
                merged[key] = "Isointense"
            else:
                merged[key] = "null"
        elif key in degree_fields:
            merged[key] = "Present"
        else:
            merged[key] = "null"

    return merged


def ensure_min_prediction_struct(pred):
    pred = pred if isinstance(pred, dict) else {}
    meta = pred.get("meta_info", {})
    signs = pred.get("structured_signs", {})
    meta = meta if isinstance(meta, dict) else {}
    signs = signs if isinstance(signs, dict) else {}
    return {
        "meta_info": {
            "predicted_modality": meta.get("predicted_modality", "null"),
            "predicted_location": meta.get("predicted_location", "null"),
            "lesion_found": meta.get("lesion_found", "False"),
        },
        "structured_signs": {k: signs.get(k, "null") for k in CORE_FIELDS},
    }


def print_unified_air_report(data_list, step_name):
    tumor_samples = [i for i in data_list if str(i["fused_label"]["meta_info"]["lesion_found"]) == "True"]
    total_tumors = len(tumor_samples)
    if total_tumors == 0:
        return

    print(f"\n📊 [{step_name}] Fusion AIR Report (Tumor Samples: {total_tumors})")
    print("=" * 60)
    for field in CORE_FIELDS:
        count = sum(1 for i in tumor_samples if not is_null(i["fused_label"]["structured_signs"].get(field)))
        rate = (count / total_tumors) * 100
        print(f"    - {field.ljust(17)}: {'█' * int(rate / 5)}{'░' * (20 - int(rate / 5))} {rate:.1f}%")
    print("=" * 60 + "\n")


# ==================== Main ====================
def fuse_results(file_a: str, file_b: str, source_file: str, output_file: str, healthy_file: str):
    if not (os.path.exists(file_a) and os.path.exists(file_b)):
        raise FileNotFoundError("Missing input files: file_a or file_b")

    with open(file_a, "r", encoding="utf-8") as f:
        dict_a = {i["sample_id"]: i for i in json.load(f) if isinstance(i, dict) and i.get("sample_id")}
    with open(file_b, "r", encoding="utf-8") as f:
        dict_b = {i["sample_id"]: i for i in json.load(f) if isinstance(i, dict) and i.get("sample_id")}

    path_map = {}
    if source_file and os.path.exists(source_file):
        with open(source_file, "r", encoding="utf-8") as f:
            for i in json.load(f):
                if isinstance(i, dict) and i.get("sample_id"):
                    path_map[i["sample_id"]] = i.get("image_path", "")

    common_ids = set(dict_a.keys()) & set(dict_b.keys())
    fused_results, healthy_results = [], []

    stats = {
        "kept": 0,
        "healthy": 0,
        "dropped_single_model": 0,
        "dropped_modality_conflict": 0,
    }

    for sample_id in tqdm(common_ids, desc="Fusing A & B"):
        res_a = ensure_min_prediction_struct(dict_a[sample_id].get("ai_prediction"))
        res_b = ensure_min_prediction_struct(dict_b[sample_id].get("ai_prediction"))

        meta_a, signs_a = res_a["meta_info"], res_a["structured_signs"]
        meta_b, signs_b = res_b["meta_info"], res_b["structured_signs"]

        found_a = str(meta_a.get("lesion_found")) == "True"
        found_b = str(meta_b.get("lesion_found")) == "True"

        # 1) strict consensus filter
        if not (found_a and found_b):
            if (not found_a) and (not found_b):
                stats["healthy"] += 1
                healthy_results.append({"sample_id": sample_id, "image_path": path_map.get(sample_id, "")})
            else:
                stats["dropped_single_model"] += 1
            continue

        # 2) merge signs
        final_signs = merge_signs_with_downgrade(signs_a, signs_b)

        # 3) relaxed location (prefer A)
        final_loc = resolve_location_relaxed(meta_a.get("predicted_location"), meta_b.get("predicted_location"))

        # 4) modality resolution
        mod_a = normalize_modality(meta_a.get("predicted_modality"))
        mod_b = normalize_modality(meta_b.get("predicted_modality"))
        final_modality = resolve_modality_with_downgrade(mod_a, mod_b, final_signs)

        if final_modality == "conflict":
            stats["dropped_modality_conflict"] += 1
            continue

        # 5) final record
        stats["kept"] += 1
        fused_results.append(
            {
                "sample_id": sample_id,
                "image_path": path_map.get(sample_id, ""),
                "fused_label": {
                    "meta_info": {
                        "predicted_modality": final_modality,
                        "predicted_location": final_loc,
                        "lesion_found": "True",
                    },
                    "structured_signs": final_signs,
                },
            }
        )

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(fused_results, f, indent=2, ensure_ascii=False)

    os.makedirs(os.path.dirname(healthy_file) or ".", exist_ok=True)
    with open(healthy_file, "w", encoding="utf-8") as f:
        json.dump(healthy_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("✅ Fusion completed. Stats:")
    print(f"   🟢 Kept (Consensus Tumor):          {stats['kept']}")
    print(f"   ⚪ Excluded Healthy (Consensus):    {stats['healthy']}")
    print(f"   ⚠️  Dropped (Single Model Disagree): {stats['dropped_single_model']}")
    print(f"   ⚔️  Dropped (Modality Conflict):     {stats['dropped_modality_conflict']}")
    print("   (Location conflicts resolved by preference rule)")
    print("=" * 60)

    print_unified_air_report(fused_results, "Step 2 High-Precision Fusion")


def main():
    parser = argparse.ArgumentParser(description="Fuse Run A/B silver labels (open-source safe).")
    parser.add_argument(
        "--file_a",
        type=str,
        default=os.getenv("FILE_A", "./outputs/silver_label_extract_run_a.json"),
        help="Run A output JSON. Default: $FILE_A or ./outputs/silver_label_extract_run_a.json",
    )
    parser.add_argument(
        "--file_b",
        type=str,
        default=os.getenv("FILE_B", "./outputs/silver_label_extract_run_b.json"),
        help="Run B output JSON. Default: $FILE_B or ./outputs/silver_label_extract_run_b.json",
    )
    parser.add_argument(
        "--source_file",
        type=str,
        default=os.getenv("SOURCE_FILE", "./outputs/Dataset_For_LLM_Inference.json"),
        help="Source file mapping sample_id -> image_path. Default: $SOURCE_FILE or ./outputs/Dataset_For_LLM_Inference.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.getenv("OUTPUT_FILE", "./outputs/Fused_Silver_Labels_HIGH_PRECISION.json"),
        help="Fused output JSON. Default: $OUTPUT_FILE or ./outputs/Fused_Silver_Labels_HIGH_PRECISION.json",
    )
    parser.add_argument(
        "--healthy_out",
        type=str,
        default=os.getenv("HEALTHY_FILE", "./outputs/Excluded_Healthy_Samples.json"),
        help="Excluded healthy output JSON. Default: $HEALTHY_FILE or ./outputs/Excluded_Healthy_Samples.json",
    )
    args = parser.parse_args()

    fuse_results(args.file_a, args.file_b, args.source_file, args.output, args.healthy_out)


if __name__ == "__main__":
    main()
