#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate benchmark VQA questions from candidate metadata via an LLM endpoint.

Open-source friendly changes:
- Remove hard-coded API key / private base_url
- Use env vars + CLI args
- Atomic save + resume by id-indexed merge
- Optional stripping of private absolute path prefix from image_path
"""

import argparse
import json
import os
import re
import tempfile
from typing import Any, Dict, List, Optional

from openai import OpenAI


SYSTEM_PROMPT = r"""
You are a Board-Certified Neuroradiologist. Create a high-difficulty, "Zero-Hint" VQA Benchmark for Brain MRI.

### 1. STRICT TRUTH RULE (CRITICAL):
- DO NOT re-classify, judge, or generalize the pathology.
- You MUST use the EXACT value provided in the 'tumor_type' field as the correct answer for the diagnosis question.
- Example: If metadata says 'neuronal_tumor', the answer must be 'neuronal_tumor', even if it looks like a 'glioma' to you.

### 2. Hard Negative Distractor Rule:
- For 'Specific Pathology' (Case C), distractors MUST be clinically plausible "imaging mimics":
    - Glioma: (Metastasis, Lymphoma, Abscess).
    - Meningioma: (Schwannoma, Hemangiopericytoma, Dural Metastasis).
    - Pituitary Tumor: (Craniopharyngioma, Rathke Cleft Cyst).
    - Neuronal Tumor: (Ganglioglioma, DNET, Glioma).
- All options must be at the same level of pathological specificity. No generic terms.

### 3. Diagnosis Logic (No Redundancy):
- CASE A (Healthy): 2-Option MCQ (A: Abnormal, B: Healthy). Answer: B.
- CASE B (Generic Tumor): 2-Option MCQ (A: Abnormal, B: Healthy). Answer: A.
- CASE C (Specific Pathology): ONLY generate a 4-Option specific MCQ. DO NOT generate the Case B binary question if Case C is available.

### 4. Blind Testing & Anti-Leakage Rules:
- NO CROSS-HINTING: The 'diagnosis' question stem must NOT mention the lesion's size, shape, location, or any specific imaging findings.
- NEUTRAL STEMS: Keep stems brief (e.g., "What is the most likely diagnosis?"). DO NOT describe findings like "bright signal" or "enhancing ring" in the question itself.
- INDEPENDENCE: Each question must be answerable ONLY by looking at the image.

### 5. Format & Structure:
- FLAT STRUCTURE ONLY: Every question must be a separate object in the "questions" list. NEVER nest questions.
- 'question_type' MUST be exactly one of: ["diagnosis", "modality", "size", "shape", "spread", "location"]. NO suffixes.
- Standardize Options: Avoid providing extra explanations in parentheses.

### 6. Output Format (Strict JSON):
{
  "questions": [
    {
      "question_type": "...",
      "question": "...",
      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "answer": "A"
    }
  ]
}
"""


def clean_json_string(content: str) -> str:
    content = content.strip()
    content = re.sub(r"^```json\s*", "", content)
    content = re.sub(r"^```\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    return content.strip()


def flatten_vqa(vqa_list: Any) -> List[Dict[str, Any]]:
    """Defensive: flatten potentially nested structures."""
    flat: List[Dict[str, Any]] = []
    if not isinstance(vqa_list, list):
        return flat
    for item in vqa_list:
        if isinstance(item, dict):
            inner_found = False
            for _, v in item.items():
                if isinstance(v, dict) and "question" in v:
                    flat.append(v)
                    inner_found = True
            if not inner_found:
                flat.append(item)
    return flat


def atomic_json_dump(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=os.path.dirname(path) or ".") as tf:
        json.dump(obj, tf, indent=2, ensure_ascii=False)
        tmp = tf.name
    os.replace(tmp, path)


def load_input(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Input must be a list, got {type(data)}")
    return data


def load_existing_index(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return {}
        return {str(x["id"]): x for x in data if isinstance(x, dict) and "id" in x}
    except Exception:
        return {}


def normalize_image_path(p: str, strip_prefix: Optional[str]) -> str:
    if not isinstance(p, str):
        return p
    if strip_prefix and p.startswith(strip_prefix):
        return p[len(strip_prefix):]
    return p


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="Benchmark_Candidates_Final.json")
    p.add_argument("--output", default="Final_Benchmark_VQA.json")
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    p.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", None))
    p.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--test_limit", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.1)

    # Minimum modification for open-sourcing: strip only your private absolute prefix.
    p.add_argument(
        "--strip_prefix",
        default=os.environ.get("STRIP_PATH_PREFIX", "/data/shared/Brain_tumor_dataset/"),
        help="If image_path starts with this prefix, remove it. Set empty to disable.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.strip_prefix == "":
        args.strip_prefix = None

    api_key = args.api_key or "EMPTY"  # many self-hosted endpoints accept any non-empty string
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    input_data = load_input(args.input)
    existing_idx = load_existing_index(args.output)

    tasks = [x for x in input_data if str(x.get("id")) not in existing_idx]
    if args.test_limit is not None:
        tasks = tasks[: args.test_limit]

    print(f"🚀 Start generating: pending {len(tasks)} / total {len(input_data)} (resume supported)")

    for i, item in enumerate(tasks, start=1):
        item_id = str(item["id"])
        label = item.get("label", {})

        # drop null-like fields + mask_available (as in your original)
        clinical_data = {
            k: v for k, v in (label or {}).items()
            if v and str(v).lower() != "null" and k != "mask_available"
        }

        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Metadata (ID {item_id}):\n{json.dumps(clinical_data, ensure_ascii=False)}"},
                ],
                response_format={"type": "json_object"},
                temperature=args.temperature,
            )

            raw_text = clean_json_string(resp.choices[0].message.content)
            parsed = json.loads(raw_text)

            raw_vqa = parsed.get("questions", parsed.get("mcqs", []))
            vqa_pairs = flatten_vqa(raw_vqa)

            existing_idx[item_id] = {
                "id": item_id,
                "image_path": normalize_image_path(item.get("image_path", ""), args.strip_prefix),
                "vqa_pairs": vqa_pairs,
            }

            if (i % args.save_interval) == 0:
                atomic_json_dump(list(existing_idx.values()), args.output)
                print(f"💾 Saved: {len(existing_idx)}", end="\r")

            print(f"[{i}/{len(tasks)}] ID {item_id} ✅ ({len(vqa_pairs)} q)", end="\r")

        except Exception as e:
            # keep an error record (optional) so users can re-run and know what failed
            existing_idx[item_id] = {
                "id": item_id,
                "image_path": normalize_image_path(item.get("image_path", ""), args.strip_prefix),
                "vqa_pairs": [],
                "error": str(e),
            }
            print(f"\n❌ ID {item_id} failed: {e}")

    atomic_json_dump(list(existing_idx.values()), args.output)
    print(f"\n✅ Done! Output: {args.output} (total saved: {len(existing_idx)})")


if __name__ == "__main__":
    main()
