import os
import json
import base64
import time
import io
import argparse
import hashlib
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from typing import Any, Dict, Optional, Tuple

# ==================== 🧠 SYSTEM PROMPT (UNCHANGED) ====================
SYSTEM_PROMPT = r"""
You are an experienced Neuroradiologist Auditor.
**Task:** "Sanity Check" Step 2 Labels against the MRI slice.
**Mindset:** Step 2 labels are your baseline. Be a conservative auditor. Only correct CLEAR and GROSS errors. 

### 1. Audit Logic (Strict Decision Tree)

#### A. Location (Threshold: "Gross Violation Only")
* **Rule:** Location is binary (Right or Wrong). No middle ground.
* **KEEP:** If there is ANY overlap, proximity, or the lesion is small/faint at the site.
* **ERASE:** ONLY for Gross Geographical Errors (e.g., Label says "Left Hemisphere", Image shows lesion on "Right").
* **Note:** Do NOT downgrade location.

#### B. Modality (Physics Check - CRITICAL)
* **Goal:** Verify if the "Predicted Modality" matches the image physics.
* **T1 vs T1CE (Contrast) Differentiation:**
    * **LOOK FOR:** 1. **Nasal Turbinates/Mucosa** (Best Indicator), 2. Dural Venous Sinuses (e.g., Sagittal Sinus), 3. Choroid Plexus.
    * **T1 (Non-Contrast):** These structures appear **DARK** or Isointense.
    * **T1CE (Contrast):** These structures MUST be **BRIGHT WHITE** (Hyperintense).
* **ACTIONS:**
    * **KEEP:** If physics match or image is ambiguous.
    * **DOWNGRADE:** If label is "T1_Contrast" BUT nasal turbinates/sinuses are clearly DARK -> Change to "T1".
    * **ERASE:** If label is "T2" but CSF is DARK (physically impossible).

#### C. Signs (Visual Cleaning & De-specification)
* **Goal:** Validate specific visual features.
* **KEEP:** If the sign is visible OR suggestive. (Benefit of the doubt).
* **ERASE:** Only if the sign is **100% absent** (Hallucination).
* **DOWNGRADE (De-specification):**
    * **Definition:** The feature exists (True Positive), but the Step 2 description is **too specific** or **wrongly detailed**.
    * **Logic:** "I see the abnormality, but I don't see *that specific* pattern."
    * **Examples:**
        * Label="Ring-Enhancing" -> Image shows blurry white blob -> Action: **DOWNGRADE** (to generic "Enhancing").
        * Label="Spiculated Margins" -> Image shows fuzzy edges -> Action: **DOWNGRADE** (to generic "Indistinct").
        * Label="Necrotic Center" -> Image is solid gray -> Action: **DOWNGRADE** (to generic "Heterogeneous").

### 2. Output Format (FULL JSON REQUIRED)
{
  "final_decision": "ACCEPT",
  "field_actions": {
    "predicted_modality": "KEEP|ERASE|DOWNGRADE",
    "predicted_location": "KEEP|ERASE",
    "shape": "KEEP|ERASE|DOWNGRADE",
    "margins": "KEEP|ERASE|DOWNGRADE",
    "texture": "KEEP|ERASE|DOWNGRADE",
    "enhancement": "KEEP|ERASE|DOWNGRADE",
    "edema": "KEEP|ERASE|DOWNGRADE",
    "signal_intensity": "KEEP|ERASE"
  },
  "additional_findings": "Max 3 words (e.g., 'Artifact present', 'Wrong orientation'), null if none."
}
"""

CORE_FIELDS = ["shape", "margins", "texture", "enhancement", "edema", "signal_intensity"]


# ==================== 🧩 Utils ====================
def is_null(v) -> bool:
    if v is None:
        return True
    return str(v).lower().strip() in ["null", "none", "unknown", "", "nan"]


def stable_short_id(text: str, n: int = 10) -> str:
    if not text:
        return "noid"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def load_json(path: str):
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_json_atomic(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def build_source_map(source_file: str):
    data = load_json(source_file)
    return {
        item.get("sample_id"): {
            "true_tumor_type": item.get("tumor_type", "Unknown"),
            "true_modality": item.get("modality", "Unknown"),
        }
        for item in data
        if isinstance(item, dict) and item.get("sample_id")
    }


def read_image(dataset_root: str, rel_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Image normalization: thumbnail into 512x512 canvas (center pad).
    Keeps aspect, avoids distortion for vision model input stability.
    """
    rel = (rel_path or "").replace("\\", "/").lstrip("/")
    full_path = os.path.join(dataset_root, rel)
    if not os.path.exists(full_path):
        return None, None

    try:
        with Image.open(full_path) as img:
            img = img.convert("RGB")
            img.thumbnail((512, 512), Image.Resampling.LANCZOS)
            new_img = Image.new("RGB", (512, 512), (0, 0, 0))
            new_img.paste(img, ((512 - img.size[0]) // 2, (512 - img.size[1]) // 2))

            buf = io.BytesIO()
            new_img.save(buf, format="JPEG", quality=95)
            b = buf.getvalue()
            return base64.b64encode(b).decode("utf-8"), "image/jpeg"
    except Exception:
        return None, None


def extract_content_text(choice_message_content: Any) -> Optional[str]:
    """
    OpenAI content may be str or a list of parts.
    """
    if choice_message_content is None:
        return None
    if isinstance(choice_message_content, str):
        return choice_message_content
    if isinstance(choice_message_content, list):
        parts = []
        for x in choice_message_content:
            if isinstance(x, dict) and "text" in x:
                parts.append(x["text"])
            else:
                parts.append(str(x))
        return "\n".join(parts).strip() or None
    return str(choice_message_content).strip() or None


def extract_json_robust(content: Any) -> Dict[str, Any]:
    text = extract_content_text(content)
    if not text:
        return {}
    text = text.strip()

    # strip code fences if any
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    # try parse whole
    try:
        return json.loads(text)
    except Exception:
        pass

    # try parse substring between first { and last }
    try:
        s = text.find("{")
        e = text.rfind("}") + 1
        if s >= 0 and e > s:
            return json.loads(text[s:e])
    except Exception:
        return {}

    return {}


def print_final_air_report(accepted_list):
    total = len(accepted_list)
    if total == 0:
        return
    print("\n" + "=" * 60)
    print("📊 [Step 3] Final Gold Standard AIR Report")
    print("=" * 60)
    field_rates = []
    for field in CORE_FIELDS:
        count = sum(1 for i in accepted_list if not is_null(i["gold_label"]["structured_signs"].get(field)))
        rate = (count / total) * 100
        field_rates.append(rate)
        bar = "█" * int(rate / 5) + "░" * (20 - int(rate / 5))
        print(f"    - {field.ljust(17)}: {bar} {rate:.1f}%")
    print("-" * 60)
    print(f"    🌟 Final Average Information Rate (AIR): {np.mean(field_rates):.2f}%")
    print(f"    🎯 Qualified Samples: {total}")
    print("=" * 60 + "\n")


# ==================== 🚀 Audit Unit ====================
def audit_one(
    client: OpenAI,
    model_name: str,
    temperature: float,
    dataset_root: str,
    item: Dict[str, Any],
    source_map: Dict[str, Dict[str, Any]],
    log_sample_id: bool,
):
    sid = (item.get("sample_id") or "").strip()
    sid_show = sid if log_sample_id else stable_short_id(sid)

    label = item.get("fused_label", {}) or {}
    meta = label.get("meta_info", {}) or {}
    signs = label.get("structured_signs", {}) or {}

    img_b64, mime = read_image(dataset_root, item.get("image_path", ""))
    if not img_b64:
        return {"status": "REJECT", "sample_id": sid, "reason": "image_missing"}

    user_payload = {
        "modality": meta.get("predicted_modality"),
        "location": meta.get("predicted_location"),
        "signs": signs,
    }
    user_content = json.dumps(user_payload, ensure_ascii=False)

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Audit these labels: {user_content}"},
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                        ],
                    },
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
                timeout=45.0,
            )

            out = extract_json_robust(resp.choices[0].message.content)
            actions = out.get("field_actions", {}) if isinstance(out, dict) else {}
            if not isinstance(actions, dict):
                actions = {}

            final_label_meta = dict(meta)
            final_signs = dict(signs)
            logs = []

            # 1) Modality
            mod_act = str(actions.get("predicted_modality", "KEEP")).upper()
            if mod_act == "ERASE":
                final_label_meta["predicted_modality"] = None
                logs.append("🧬 Mod->null")
            elif mod_act == "DOWNGRADE":
                if "contrast" in str(meta.get("predicted_modality", "")).lower():
                    final_label_meta["predicted_modality"] = "T1"
                    logs.append("🧬 Mod:CE->T1")

            # 2) Location
            if str(actions.get("predicted_location", "KEEP")).upper() == "ERASE":
                final_label_meta["predicted_location"] = None
                logs.append("📍 Loc->null")

            # 3) Signs
            force_no_enhancement = (
                final_label_meta.get("predicted_modality") == "T1"
                and "contrast" in str(meta.get("predicted_modality", "")).lower()
            )

            for k in CORE_FIELDS:
                if is_null(signs.get(k)):
                    continue

                if force_no_enhancement and k == "enhancement":
                    final_signs[k] = None
                    logs.append("🔧 enhancement->null")
                    continue

                act = str(actions.get(k, "KEEP")).upper()
                if act == "ERASE":
                    final_signs[k] = None
                    logs.append(f"🔧 {k}->null")
                elif act == "DOWNGRADE":
                    if k in ["enhancement", "edema"]:
                        final_signs[k] = "Present"
                    # NOTE: signal_intensity is NO DOWNGRADE in prompt; keep as-is
                    logs.append(f"🔧 {k}->DOWN")

            src_info = source_map.get(sid, {})
            gold_label = {
                "meta_info": {**final_label_meta, **src_info},
                "structured_signs": final_signs,
                "additional_clues": out.get("additional_findings") if not is_null(out.get("additional_findings")) else None,
            }

            return {
                "status": "ACCEPT",
                "sample_id": sid,
                "image_path": item.get("image_path"),
                "gold_label": gold_label,
                "logs": logs,
            }

        except Exception as e:
            if attempt == 2:
                return {"status": "REJECT", "sample_id": sid, "reason": f"api_error:{str(e)}"}
            time.sleep(2 ** (attempt + 1))

    return {"status": "REJECT", "sample_id": sid, "reason": "max_attempts_reached"}


# ==================== 🚀 Main ====================
def main():
    parser = argparse.ArgumentParser(description="Step 3 QC audit for fused labels (open-source safe).")
    parser.add_argument("--model", type=str, default=os.getenv("MODEL_NAME", "gemini-3-flash-preview"))
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--log_sample_id", action="store_true", help="Print raw sample_id in logs (default hashed).")

    parser.add_argument(
        "--dataset_root",
        type=str,
        default=os.getenv("DATASET_ROOT", "./data"),
        help="Dataset root containing image files. Default: $DATASET_ROOT or ./data",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.getenv("INPUT_FILE", "./outputs/Fused_Silver_Labels_HIGH_PRECISION.json"),
        help="Fused labels JSON. Default: $INPUT_FILE or ./outputs/Fused_Silver_Labels_HIGH_PRECISION.json",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=os.getenv("SOURCE_FILE", "./outputs/all_brain_tumor_metadata.json"),
        help="Metadata source JSON for tumor_type/modality. Default: $SOURCE_FILE or ./outputs/all_brain_tumor_metadata.json",
    )
    parser.add_argument(
        "--accept_out",
        type=str,
        default=os.getenv("ACCEPT_FILE", "./outputs/Final_Gold_Standard_CLEAN.json"),
        help="Accepted gold output JSON. Default: $ACCEPT_FILE or ./outputs/Final_Gold_Standard_CLEAN.json",
    )
    parser.add_argument(
        "--reject_out",
        type=str,
        default=os.getenv("REJECT_FILE", "./outputs/QC_Rejected_Gemini.json"),
        help="Rejected output JSON. Default: $REJECT_FILE or ./outputs/QC_Rejected_Gemini.json",
    )
    args = parser.parse_args()

    # Secrets from env ONLY
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Refusing to run to avoid hardcoded secrets.")
    base_url = os.getenv("OPENAI_BASE_URL")  # optional
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    source_map = build_source_map(args.source)
    all_fused = load_json(args.input)
    if not all_fused:
        print("❌ Empty input; nothing to audit.")
        return

    accepted_gold = load_json(args.accept_out)
    rejected = load_json(args.reject_out)

    done_ids = {str(i.get("sample_id")).strip() for i in accepted_gold if i.get("sample_id")}
    fail_ids = {str(i.get("sample_id")).strip() for i in rejected if i.get("sample_id")}

    to_process = [
        i
        for i in all_fused
        if str(i.get("sample_id")).strip() not in done_ids and str(i.get("sample_id")).strip() not in fail_ids
    ]

    if not to_process:
        print("✅ All tasks completed.")
        print_final_air_report(accepted_gold)
        return

    print(f"🚀 Step 3 Final Audit | Tasks: {len(to_process)} | workers={args.max_workers}")

    processed = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {
            ex.submit(
                audit_one,
                client,
                args.model,
                args.temperature,
                args.dataset_root,
                item,
                source_map,
                args.log_sample_id,
            ): item
            for item in to_process
        }

        for f in tqdm(as_completed(futures), total=len(to_process), desc="Auditing"):
            r = f.result()
            sid = (r.get("sample_id") or "").strip()
            sid_show = sid if args.log_sample_id else stable_short_id(sid)

            if r.get("status") == "ACCEPT":
                accepted_gold.append(r)
                if r.get("logs"):
                    tqdm.write(f"✅ [Update] {sid_show}: {', '.join(r['logs'])}")
            else:
                rejected.append(r)
                tqdm.write(f"❌ [Reject] {sid_show}: {r.get('reason')}")

            processed += 1
            if processed % args.save_interval == 0:
                save_json_atomic(accepted_gold, args.accept_out)
                save_json_atomic(rejected, args.reject_out)

    save_json_atomic(accepted_gold, args.accept_out)
    save_json_atomic(rejected, args.reject_out)
    print_final_air_report(accepted_gold)


if __name__ == "__main__":
    main()
