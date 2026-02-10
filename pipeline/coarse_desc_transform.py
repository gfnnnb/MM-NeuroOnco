import json
import os
import argparse
from tqdm import tqdm

UNKNOWN_LABELS = {"unknown", "none", "null", "nan", "undefined"}
HEALTHY_LABELS = {"notumor", "no tumor", "healthy", "normal", "control"}


def generate_description(semantics, tumor_type, modality):
    # --- 1. Clean inputs ---
    if tumor_type is not None:
        t_type_str = str(tumor_type).strip()
        t_type_display = t_type_str.title()
        t_type_lower = t_type_str.lower()
    else:
        t_type_display = "Unknown"
        t_type_lower = "unknown"

    if modality is not None:
        mod_str = str(modality).strip()
        mod_upper = mod_str.upper()
        mod_lower = mod_str.lower()
    else:
        mod_upper = None
        mod_lower = "unknown"

    # --- 2. Base sentence ---
    if mod_upper and mod_lower not in UNKNOWN_LABELS:
        base_str = f"A {mod_upper} brain MRI scan"
        modality_note = ""
    else:
        base_str = "A brain MRI scan"
        modality_note = " The specific imaging modality is unknown."

    # --- 3. Healthy branch ---
    if t_type_lower in HEALTHY_LABELS:
        desc = f"{base_str}. The scan appears normal with no visible pathological findings."
        if modality_note:
            desc += modality_note
        return desc

    # --- 4. Tumor / abnormal branch ---
    if t_type_lower in UNKNOWN_LABELS:
        pathology_desc = f"{base_str} showing an abnormal mass."
        type_note = " The specific tumor type is not labeled."
    elif t_type_lower in {"brain tumor", "tumor"}:
        pathology_desc = f"{base_str} showing a brain tumor."
        type_note = " The specific histological subtype is not specified."
    else:
        pathology_desc = f"{base_str} showing signs of {t_type_display}."
        type_note = ""

    desc = pathology_desc + modality_note + type_note

    # morphology details
    if semantics and isinstance(semantics, dict):
        size = str(semantics.get("size", "unknown size")).lower()
        loc = str(semantics.get("location", "unknown location")).lower()
        shape = str(semantics.get("shape", "irregular")).lower()
        spread = str(semantics.get("spread", "lesion")).lower()

        spread_phrase = f"a {spread}" if "solitary" in spread else spread

        desc += (
            f" The mass is {size} and located in the {loc}. "
            f"It presents as a {shape} structure and appears as {spread_phrase}."
        )
    else:
        desc += (
            " No segmentation mask is available, "
            "so detailed morphological features (size, shape, location) cannot be determined."
        )

    return desc


def transform_to_llm_format(input_json: str, output_json: str) -> None:
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Input file not found: {input_json}")

    print(f"📂 Loading: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of items.")

    light_dataset = []
    print("✂️ Generating coarse descriptions (using existing tumor_type/modality)...")

    for item in tqdm(data):
        sample_id = item.get("sample_id")
        image_path = item.get("image_path")

        tumor_type = item.get("tumor_type")
        modality = item.get("modality")
        semantics = item.get("medical_semantics")

        coarse_desc = generate_description(semantics, tumor_type, modality)

        light_dataset.append(
            {
                "sample_id": sample_id,
                "image_path": image_path,
                "coarse_description": coarse_desc,
            }
        )

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    print(f"\n💾 Saving: {output_json}")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(light_dataset, f, indent=2, ensure_ascii=False)

    print(f"✅ Done. Total: {len(light_dataset)}")


def main():
    parser = argparse.ArgumentParser(description="Transform rich metadata into LLM inference JSON.")
    parser.add_argument(
        "--input",
        type=str,
        default=os.getenv("INPUT_RICH_FILE", "./outputs/all_brain_tumor_metadata_rich.json"),
        help="Input rich metadata JSON. Default: $INPUT_RICH_FILE or ./outputs/all_brain_tumor_metadata_rich.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.getenv("OUTPUT_LLM_FILE", "./outputs/Dataset_For_LLM_Inference.json"),
        help="Output JSON for LLM inference. Default: $OUTPUT_LLM_FILE or ./outputs/Dataset_For_LLM_Inference.json",
    )
    args = parser.parse_args()

    transform_to_llm_format(args.input, args.output)


if __name__ == "__main__":
    main()
