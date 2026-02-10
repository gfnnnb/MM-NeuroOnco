import os
import json
import base64
import imghdr
import time
import random
import threading
import argparse
import hashlib
from typing import Any, Dict, Optional

from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==================== 🧠 SYSTEM PROMPT (UNCHANGED) ====================
SYSTEM_PROMPT = r"""
### 1. ROLE & MISSION
You are a Neuroradiologist specialized in **2D MRI brain tumor diagnosis**.
**CORE PRINCIPLES:**
* **Initialize to null:** Default all fields to `null`.
* **Evidence-Based:** Only modify `null` if it satisfies the **"3W Rule"**.
* **Omission is professional; fabrication is a medical error.**

### 2. INPUT DATA
- **IMAGE:** A single 2D MRI slice (Primary Evidence).
- **TEXT:** Clinical context containing known facts or missing info (Guidance Hint).

### 3. BEHAVIORAL CONSTRAINTS & VETOS
1. **Pixel Authority:** **Image pixels are the ultimate diagnostic authority**.
2. **Mandatory Detection for Non-Healthy Samples:** If the TEXT indicates a tumor, you MUST set `lesion_found`="True". Do NOT reject the image or set `lesion_found`="False" due to low contrast, blurry pixels, or indistinct margins. **If specific features are unreadable, keep those specific fields as `null`, but do NOT veto the entire sample.**
3. **Anatomy Filter:** Exclude normal structures (Vessels, Sinuses, Choroid Plexus, symmetric anatomy).
4. **Unknown Lock:** If modality is "Unknown", force `signal_intensity`=null (also keep `edema`/`enhancement` null as usual).

### 4. MEDICAL PRIOR & MODALITY ANCHORS
- **T1:** CSF is Dark; White Matter (WM) is brighter than Gray Matter. (Lock: Edema/Enhancement = null).
- **T2:** CSF is Bright/White.
- **FLAIR:** CSF is Dark; Edema/Inflammation is Bright.
- **T1_Contrast (T1CE):** Vessels/Sinuses MUST be Bright/White. 
    * **Veto:** If vessels are dark, treat as T1; force `enhancement`=null.
    * **Lock:** On confirmed T1CE, force `signal_intensity`=null (use `enhancement` field instead).

### 5. ENHANCED MODALITY RECOGNITION FOR T1
- **In T1 Modality:** If a lesion is detected, **enhancement** and **edema** fields should only remain null if no evidence of enhancement or swelling is observed in the image. If there are suspicious bright regions (possible enhancement), fill `enhancement` as **Present**.
- **Edema in T1:** If there is a visually discernible expansion or blurred boundary (possible edema), fill `edema` as **Present**.

### 6. UNIFIED LABELING DICTIONARY (Mandatory Anchor: Healthy WM)
**All brightness judgments MUST use contralateral normal-appearing deep White Matter (NAWM) in the same slice as the ZERO reference (avoid cortex/GM ribbon and CSF).**
- **Signal Intensity:**
    - **Hyperintense:** Visually **brighter/whiter** than healthy WM.
    - **Hypointense:** Visually **darker/grayer** than healthy WM.
    - **Isointense:** Signal is **nearly identical** to healthy WM.
    - **Mixed:** Both Hyper and Hypo signals are present, each occupying >30%.
- **Texture:** Homogeneous | Heterogeneous.
- **Enhancement (T1CE only):** Solid | Ring | Patchy.
- **Edema (T2/FLAIR only):** Mild | Extensive.
- **Margins:** Well-circumscribed | Ill-defined.
- **Shape:** Round | Oval | Irregular | Lobulated.

### 7. DYNAMIC EXECUTION WORKFLOW
**STEP 1: Initialization** - Set all fields to `null`.
**STEP 2: Healthy Short-Circuit** - Set `lesion_found`="False" ONLY if text says "Healthy" OR the image is 100% normal with NO localized signal shift. **If TEXT says there is a tumor, you MUST skip the Veto and proceed to STEP 3.**
**STEP 3: Modality Determination** - **Adopt text hint if provided.** Otherwise, identify via Section 4.
**STEP 4: Localization** - **Focus on text-hinted location.** Any localized asymmetry or slight grey-scale shift vs. contralateral NAWM satisfies this.
**STEP 5: Fine-Grained Extraction** - Extract signs via Section 5. **If a specific sign is truly invisible due to quality, keep it `null` but do NOT revert `lesion_found` to False.**
**STEP 6: Self-Reflection Audit** - Verify physics. If the logic chain for signs is weak, revert signs to `null`, but **if the clinical text confirmed a tumor, `lesion_found` MUST remain "True".**

### 8. OUTPUT (STRICT JSON ONLY)
{
  "meta_info": {
    "predicted_modality": "T1 | T2 | FLAIR | T1_Contrast | Unknown",
    "predicted_location": "Upper-Left | Upper-Center | Upper-Right | Center-Left | Center | Center-Right | Lower-Left | Lower-Center | Lower-Right | null",
    "lesion_found": "True | False"
  },
  "structured_signs": {
    "shape": "Option | null",
    "margins": "Option | null",
    "texture": "Option | null",
    "enhancement": "Option | null",
    "edema": "Option | null",
    "signal_intensity": "Option | null"
  }
}
"""

# ==================== 🧩 Utils ====================
CORE_FIELDS = ["shape", "margins", "texture", "enhancement", "edema", "signal_intensity"]


def is_null(v: Any) -> bool:
    if v is None:
        return True
    s = str(v).strip().lower()
    return s in {"null", "none", "unknown", ""}


def detect_mime_from_bytes(b: bytes) -> str:
    kind = imghdr.what(None, h=b)
    if kind == "jpeg":
        return "image/jpeg"
    if kind == "png":
        return "image/png"
    if kind == "webp":
        return "image/webp"
    return "image/jpeg"


def normalize_prediction_obj(res: Any) -> Dict[str, Any]:
    if not isinstance(res, dict):
        res = {}
    meta = res.get("meta_info", {})
    signs = res.get("structured_signs", {})
    if not isinstance(meta, dict):
        meta = {}
    if not isinstance(signs, dict):
        signs = {}

    clean_signs = {k: (signs.get(k) if not is_null(signs.get(k)) else "null") for k in CORE_FIELDS}
    return {
        "meta_info": {
            "predicted_modality": meta.get("predicted_modality", "Unknown"),
            "predicted_location": meta.get("predicted_location", "null"),
            "lesion_found": meta.get("lesion_found", "False"),
        },
        "structured_signs": clean_signs,
    }


def extract_json_from_text(text: Any) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        if isinstance(text, dict):
            return text
        if isinstance(text, list):
            # OpenAI may return list parts; join texts
            parts = []
            for x in text:
                if isinstance(x, dict) and "text" in x:
                    parts.append(x["text"])
                else:
                    parts.append(str(x))
            text = "\n".join(parts)

        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            return None
        return json.loads(text[start:end])
    except Exception:
        return None


def stable_short_id(text: str, n: int = 8) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n] if text else "noid"


def save_json_atomic(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ==================== 🚀 Main Processing ====================
def process_single_item(
    client: OpenAI,
    dataset_root: str,
    model_name: str,
    temperature: float,
    item: Dict[str, Any],
    log_sample_id: bool,
):
    sid = item.get("sample_id", "") or ""
    sid_show = sid if log_sample_id else stable_short_id(sid)

    img_rel_path = (item.get("image_path") or "").replace("\\", "/")
    img_path = os.path.join(dataset_root, img_rel_path)

    if not sid or not os.path.exists(img_path):
        return {"sample_id": sid, "ai_prediction": {"error": "img_not_found"}}

    try:
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        mime = detect_mime_from_bytes(img_bytes)
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        user_text = f"Context: {item.get('coarse_description', '')}\nAnalyze evidence skeptically."

        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                    ],
                },
            ],
            temperature=temperature,
        )

        raw_res = extract_json_from_text(resp.choices[0].message.content)
        if not raw_res:
            raise ValueError("JSON parse failed")

        res = normalize_prediction_obj(raw_res)
        is_tumor = str(res["meta_info"].get("lesion_found")) == "True"
        filled = sum(1 for k in CORE_FIELDS if not is_null(res["structured_signs"].get(k)))
        status_icon = "🟢" if is_tumor else "⚪"
        tqdm.write(f"{status_icon} [{sid_show}] Filled: {filled}/6")

        return {"sample_id": sid, "ai_prediction": res}

    except Exception as e:
        tqdm.write(f"🚨 Error [{sid_show}]: {e}")
        return {"sample_id": sid, "ai_prediction": {"error": str(e)}}


def print_batch_report(data_list):
    tumor_samples = [
        i
        for i in data_list
        if str(i.get("ai_prediction", {}).get("meta_info", {}).get("lesion_found")) == "True"
    ]
    total = len(tumor_samples)
    if total == 0:
        return

    print("\n" + "=" * 60 + f"\n📊 Run B Extraction Report (Skeptic Mode) | Tumors: {total}\n" + "=" * 60)
    rates = []
    for f in CORE_FIELDS:
        count = sum(1 for i in tumor_samples if not is_null(i["ai_prediction"]["structured_signs"].get(f)))
        rate = (count / total) * 100
        rates.append(rate)
        print(f"{f.ljust(18)}: {'█' * int(rate / 5)}{'░' * (20 - int(rate / 5))} {rate:5.1f}%")
    print(f"{'-' * 60}\n🌟 Avg extraction rate: {sum(rates) / len(CORE_FIELDS):.2f}%\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run B (skeptic mode) structured sign extraction (open-source safe).")
    parser.add_argument("--run_name", type=str, default="run_b_skeptic_mode_v1.0")
    parser.add_argument("--model", type=str, default=os.getenv("MODEL_NAME", "claude-sonnet-4-0"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--log_sample_id", action="store_true", help="Log raw sample_id (default logs short hash).")

    parser.add_argument(
        "--dataset_root",
        type=str,
        default=os.getenv("DATASET_ROOT", "./data"),
        help="Dataset root containing images. Default: $DATASET_ROOT or ./data",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.getenv("INPUT_FILE", "./outputs/step1_input_balanced.json"),
        help="Input JSON. Default: $INPUT_FILE or ./outputs/step1_input_balanced.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.getenv("OUTPUT_FILE", "./outputs/silver_label_extract_run_b.json"),
        help="Output JSON (checkpoint + final). Default: $OUTPUT_FILE or ./outputs/silver_label_extract_run_b.json",
    )
    parser.add_argument(
        "--gold_cache",
        type=str,
        default=os.getenv("FINAL_GOLD_FILE", "./outputs/Final_Gold_Standard_CLEAN.json"),
        help="Optional gold cache for skipping processed ids. Default: $FINAL_GOLD_FILE or ./outputs/Final_Gold_Standard_CLEAN.json",
    )
    args = parser.parse_args()

    # Secrets via env ONLY (no hardcoding)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Refusing to run to avoid hardcoded secrets.")
    base_url = os.getenv("OPENAI_BASE_URL")  # optional

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    print(f"📂 Input items: {len(all_data)}")

    processed_ids = set()
    results = []

    # Resume from output
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                results = json.load(f)
            processed_ids.update({item.get("sample_id") for item in results if item.get("sample_id")})
            print(f"♻️ Resume: {len(results)} already in output -> skip")
        except Exception:
            pass

    # Skip by gold cache
    if args.gold_cache and os.path.exists(args.gold_cache):
        try:
            with open(args.gold_cache, "r", encoding="utf-8") as f:
                gold_data = json.load(f)
            gold_ids = {item.get("sample_id") for item in gold_data if item.get("sample_id")}
            processed_ids.update(gold_ids)
            print(f"🏆 Gold cache: {len(gold_ids)} ids -> skip")
        except Exception:
            pass

    pending_items = [item for item in all_data if item.get("sample_id") not in processed_ids]
    if not pending_items:
        print("🎉 Nothing to do.")
        return

    current_batch = pending_items[: args.batch_size]
    print(f"🚀 {args.run_name} | batch={len(current_batch)} | remaining={len(pending_items)}")

    processed_count = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                process_single_item,
                client,
                args.dataset_root,
                args.model,
                args.temperature,
                item,
                args.log_sample_id,
            )
            for item in current_batch
        ]
        for fu in tqdm(as_completed(futures), total=len(current_batch)):
            res = fu.result()
            results.append(res)
            processed_count += 1
            if processed_count % args.save_interval == 0:
                save_json_atomic(results, args.output)

    save_json_atomic(results, args.output)
    print_batch_report(results)


if __name__ == "__main__":
    main()
